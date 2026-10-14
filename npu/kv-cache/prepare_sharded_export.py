"""Shard an exported full vocabulary head without compiling an FP16 stack."""
import argparse
import json
import os
from pathlib import Path
import gguf
import onnx
import torch
from trained_probe import LMHead

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--chunk',type=int,default=49152)
a=p.parse_args()
assert 0<a.chunk<=49152
source=a.source.resolve(); out=a.output.resolve()
out.mkdir(exist_ok=False)
m=json.loads((source/'trained-manifest.json').read_text())
count=m['language_layers_total']; assert len(m['layers'])==count and m['stable_norm']
for f in source.iterdir():
    if f.is_file() and f.name not in ('trained-manifest.json',f'layer{count}.onnx',f'layer{count}-dataset.txt'):
        (out/f.name).symlink_to(os.path.relpath(f,out))
r=gguf.GGUFReader(a.model)
eps=r.fields[r.fields['general.architecture'].contents()+'.attention.layer_norm_rms_epsilon'].contents()
torch.set_num_threads(2)
head=LMHead(r,eps,True,m['norm_version']).eval()
weight=head.weight
shards=[]
for i,start in enumerate(range(0,m['vocab_size'],a.chunk)):
    end=min(start+a.chunk,m['vocab_size'])
    head.weight=weight[start:end].contiguous()
    f=out/f'layer{count+i}.onnx'
    torch.onnx.export(head,torch.ones(1,1,m['dim']),str(f),input_names=['hidden'],output_names=['logits'],opset_version=13,dynamo=False)
    model=onnx.load(f);model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,f)
    (out/f'layer{count+i}-dataset.txt').write_text((source/f'layer{count}-dataset.txt').read_text())
    shards.append({'layer':count+i,'vocab_start':start,'vocab_count':end-start})
assert sum(s['vocab_count'] for s in shards)==m['vocab_size']
m.update(head_shards=shards,head_sharding_reason='Bound output width using previously validated <=49152-row graph contract; all vocabulary rows retained')
(out/'trained-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'layers':count,'vocab':m['vocab_size'],'shards':shards}))
