"""Free generation with exported constants and FP32 arithmetic (host only).

Separates weight re-quantization damage from board execution error. This
diagnostic recomputes prefixes and is never a CPU performance baseline.
"""
import argparse,json
from pathlib import Path
import gguf,numpy as np,torch
from tokenizers import Tokenizer
from trained_probe import TrainedDecoder,LMHead
from exported_int8 import apply_exported_int8,apply_exported_head
p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--variant',required=True)
p.add_argument('--prompt',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--max-new',type=int,default=32)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
r=gguf.GGUFReader(a.model);m=json.loads((a.source/'trained-manifest.json').read_text())
ids=json.loads(a.prompt.read_text())['token_ids'];tokens=[];layers=[]
with torch.inference_mode():
    for i in range(m['language_layers_total']):
        l=TrainedDecoder(r,i).eval();apply_exported_int8(l,a.source/a.variant/f'l{i}');layers.append(l)
    head=LMHead(r,layers[0].eps).eval();apply_exported_head(head,a.source,a.variant,m)
    t=next(t for t in r.tensors if t.name=='token_embd.weight')
    emb=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy())
    for _ in range(a.max_new):
        hidden=emb[ids+tokens][None]
        for l in layers:hidden=l.full(hidden)
        token=int(head(hidden[:,-1:]).argmax());tokens.append(token)
        print(token,flush=True)
        if token in m['eos_token_ids']:break
tokenizer=Tokenizer.from_file(str(a.source/'tokenizer.json'))
a.output.write_text(json.dumps({'scope':__doc__,'variant':a.variant,'layers':len(layers),
    'prompt_ids':ids,'tokens':tokens,'answer':tokenizer.decode(tokens)},ensure_ascii=False,indent=2)+'\n')
