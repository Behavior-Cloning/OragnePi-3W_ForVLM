"""Bound vocabulary rows per NPU graph; preserve the complete output head."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import gguf
import onnx
import torch
from trained_probe import LMHead

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--chunk-size',type=int,default=49152)
args=p.parse_args();torch.set_num_threads(2)
if not 1<=args.chunk_size<=49152:raise ValueError('chunk must stay within the validated head width')
source=args.source.resolve();out=args.output.resolve();out.mkdir(exist_ok=True)
manifest=json.loads((source/'trained-manifest.json').read_text());count=len(manifest['layers'])
for path in source.iterdir():
    if not path.is_file() or path.name in ('trained-manifest.json',f'layer{count}.onnx',f'layer{count}-dataset.txt'):continue
    # Relative links survive the compiler container's /work bind mount.
    (out/path.name).symlink_to(os.path.relpath(path,out))
reader=gguf.GGUFReader(args.model)
eps=reader.fields[reader.fields['general.architecture'].contents()+'.attention.layer_norm_rms_epsilon'].contents()
head=LMHead(reader,eps,stable_norm=True,norm_version=3).eval();weight=head.weight.clone()
shards=[]
for index,start in enumerate(range(0,manifest['vocab_size'],args.chunk_size)):
    width=min(args.chunk_size,manifest['vocab_size']-start);head.weight=weight[start:start+width].contiguous()
    path=out/f'layer{count+index}.onnx'
    torch.onnx.export(head,torch.ones(1,1,manifest['dim']),str(path),input_names=['hidden'],output_names=['logits'],opset_version=13,dynamo=False)
    model=onnx.load(path);model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,path)
    (out/f'layer{count+index}-dataset.txt').write_text((source/f'layer{count}-dataset.txt').read_text())
    shards.append({'layer':count+index,'vocab_start':start,'vocab_count':width})
manifest['head_shards']=shards;manifest['head_sharding_reason']='unsharded 151674-output NPU graph returned only first 20602 nonzero logits'
(out/'trained-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
compiled=out/'fp16';compiled.mkdir()
records=json.loads((source/'fp16/stack-conversion.json').read_text())[:count]
for i in range(count):(compiled/f'l{i}').symlink_to(source/'fp16'/f'l{i}',target_is_directory=True)
def compile_head(shard):
    i=shard['layer'];folder=compiled/f'l{i}'
    with (compiled/f'layer{i}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('compile_probe.py')),
            '--source',str(out),'--output',str(folder),'--layer',str(i),'--dtype','fp16'],
            stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    record={'layer':i,'exit_code':result.returncode,'path':str(folder/'export_nbg_unify/network_binary.nb')}
    print(json.dumps(record),flush=True);return record
with ThreadPoolExecutor(max_workers=2) as pool:records+=list(pool.map(compile_head,shards))
(compiled/'stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(any(r['exit_code'] for r in records))
