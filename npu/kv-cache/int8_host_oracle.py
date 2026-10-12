"""Generate with the exact NPU INT8 decoder weights and FP32 activations on host."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from tokenizers import Tokenizer
from trained_probe import TrainedDecoder,LMHead
from exported_int8 import apply_exported_int8,apply_exported_head

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--dtype',default='w8-vdfp')
p.add_argument('--prompt-folder',type=Path,required=True);p.add_argument('--max-new',type=int,default=96)
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(2);reader=gguf.GGUFReader(args.model)
manifest=json.loads((args.source/'trained-manifest.json').read_text())
prompt=json.loads((args.prompt_folder/'prompt.json').read_text())
vocab=Tokenizer.from_file(str(args.source/'tokenizer.json'))
emb=next(t for t in reader.tensors if t.name=='token_embd.weight')
embedding=torch.from_numpy(gguf.dequantize(emb.data,emb.tensor_type).copy())
capacity=manifest['capacity'];layers=[];weights=[]
with torch.inference_mode():
    for i in manifest['layers']:
        layer=TrainedDecoder(reader,i).eval()
        records=apply_exported_int8(layer,args.source/args.dtype/f'l{i}')
        layers.append(layer);weights.append({'layer':i,'weights':records})
    head=LMHead(reader,layers[0].eps).eval()
    weights.append({'head':apply_exported_head(head,args.source,args.dtype,manifest)})
    cache=torch.zeros(len(layers),2,1,manifest['kv_heads'],capacity,manifest['head_dim'])
    generated=[];tokens=prompt['token_ids'];eos=manifest['eos_token_ids']
    for pos in range(min(capacity,len(tokens)+args.max_new-1)):
        token=tokens[pos] if pos<len(tokens) else generated[-1]
        x=embedding[token][None,None];mask=torch.full((1,1,1,capacity+1),-10000.)
        mask[...,:pos]=0;mask[...,-1]=0
        cos,sin=layers[0].rope(torch.tensor([pos]))
        for i,layer in enumerate(layers):
            x,k,v=layer(x,cache[i,0],cache[i,1],mask,cos,sin)
            cache[i,0,:,:,pos:pos+1]=k;cache[i,1,:,:,pos:pos+1]=v
        if pos>=len(tokens)-1:
            generated.append(int(head(x).argmax()))
            if generated[-1] in eos:break
report={'scope':'host oracle, exported decoder/head matrices, biases and norm scales, FP32 arithmetic/activations/KV; no board performance claim',
        'prompt':prompt,'tokens':generated,'answer':vocab.decode(generated,skip_special_tokens=False),
        'stopped_eos':generated[-1] in eos,'weights':weights}
args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('weights','prompt')},ensure_ascii=False,indent=2))
