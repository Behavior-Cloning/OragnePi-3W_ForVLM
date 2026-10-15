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
p.add_argument('--original-head',action='store_true')
p.add_argument('--original-decoder',action='store_true')
p.add_argument('--restore-matrices',nargs='*',choices=['q','k','v','o','gate','up','down'],default=[])
p.add_argument('--value-dfp-bits',type=int,choices=[8],help='host hypothesis: quantize original V using symmetric peak DFP')
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
r=gguf.GGUFReader(a.model);m=json.loads((a.source/'trained-manifest.json').read_text())
ids=json.loads(a.prompt.read_text())['token_ids'];tokens=[];layers=[]
with torch.inference_mode():
    for i in range(m['language_layers_total']):
        l=TrainedDecoder(r,i).eval()
        saved={name:getattr(l,name).clone() for name in a.restore_matrices}
        original_v=l.v.clone() if a.value_dfp_bits else None
        if not a.original_decoder:apply_exported_int8(l,a.source/a.variant/f'l{i}')
        for name,value in saved.items():getattr(l,name).copy_(value)
        if a.value_dfp_bits:
            assert not a.restore_matrices
            limit=2**(a.value_dfp_bits-1)-1
            fl=int(np.floor(np.log2(limit/float(original_v.abs().max()))))
            l.v.copy_((original_v*2.**fl).round().clamp(-limit,limit)*2.**(-fl))
        layers.append(l)
    head=LMHead(r,layers[0].eps).eval()
    if not a.original_head:apply_exported_head(head,a.source,a.variant,m)
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
    'original_head':a.original_head,'original_decoder':a.original_decoder,'restored_matrices':a.restore_matrices,
    'value_dfp_bits_hypothesis':a.value_dfp_bits,
    'prompt_ids':ids,'tokens':tokens,'answer':tokenizer.decode(tokens)},ensure_ascii=False,indent=2)+'\n')
