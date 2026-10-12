"""Separate local layer error from accumulated error using actual NPU hidden history."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from trained_probe import TrainedDecoder
from exported_int8 import apply_exported_int8

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--dtype',required=True)
p.add_argument('--case');args=p.parse_args();torch.set_num_threads(2)
manifest=json.loads((args.source/'trained-manifest.json').read_text())
board=args.source/args.dtype/('board'+('-'+args.case if args.case else ''))
prompt=json.loads((board/'prompt.json').read_text());result=json.loads((board/'tokens.json').read_text())
tokens=prompt['token_ids']+result['tokens'][:-1]
trace=np.fromfile(board/'hidden-trace.f32',np.float32).reshape(len(tokens),manifest['language_layers_total'],manifest['dim'])
reader=gguf.GGUFReader(args.model);embedding=next(t for t in reader.tensors if t.name=='token_embd.weight')
inputs=gguf.dequantize(embedding.data,embedding.tensor_type)[tokens]
rows=[]
with torch.inference_mode():
    for index in manifest['layers']:
        layer=TrainedDecoder(reader,index).eval();apply_exported_int8(layer,args.source/args.dtype/f'l{index}')
        reference=layer.full(torch.from_numpy(inputs[None].copy())).numpy()[0]
        actual=trace[:,index];step=np.linalg.norm(actual-reference,axis=-1)/np.maximum(np.linalg.norm(reference,axis=-1),1e-20)
        worst=np.argsort(step)[-5:][::-1]
        rows.append({'layer':index,'relative_l2':float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
            'worst_positions':[{'position':int(i),'token':tokens[i],'relative_l2':float(step[i]),
                'reference_max_abs':float(np.abs(reference[i]).max()),'actual_max_abs':float(np.abs(actual[i]).max())} for i in worst]})
        inputs=actual
report={'scope':'FP32 layer with exported constants and actual incoming NPU hidden trajectory; reconstructs ideal local KV, so includes cache/execution error but excludes upstream hidden drift', 'layers':rows}
(board/'local-layer-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for row in sorted(rows,key=lambda r:r['relative_l2'],reverse=True):print(json.dumps(row),flush=True)
