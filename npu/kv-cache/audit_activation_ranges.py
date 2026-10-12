"""Inspect static INT16 clipping against an independent, uncalibrated trajectory."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from trained_probe import TrainedDecoder,feed_for

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--dtype',required=True)
p.add_argument('--case');p.add_argument('--layer',type=int,default=0)
args=p.parse_args();torch.set_num_threads(2)
board=args.source/args.dtype/('board'+('-'+args.case if args.case else ''))
prompt=json.loads((board/'prompt.json').read_text());result=json.loads((board/'tokens.json').read_text())
tokens=prompt['token_ids']+result['tokens'][:-1]
reader=gguf.GGUFReader(args.model)
emb=next(t for t in reader.tensors if t.name=='token_embd.weight')
hidden=torch.from_numpy(gguf.dequantize(emb.data,emb.tensor_type)[tokens][None].copy())
with torch.inference_mode():
    for i in range(args.layer):hidden=TrainedDecoder(reader,i).eval().full(hidden)
layer=TrainedDecoder(reader,args.layer).eval()
capacity=json.loads((args.source/'trained-manifest.json').read_text())['capacity']
model=onnx.shape_inference.infer_shapes(onnx.load(args.source/f'layer{args.layer}.onnx'))
infos={v.name:v for v in list(model.graph.value_info)+list(model.graph.output)}
original_outputs=[v.name for v in model.graph.output]
names=sorted(infos)
del model.graph.output[:]
model.graph.output.extend(infos[name] for name in names)
options=ort.SessionOptions();options.intra_op_num_threads=2
session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
folder=args.source/args.dtype/f'l{args.layer}'
graph=json.loads((folder/'decoder.json').read_text())['Layers']
quant=yaml.safe_load((folder/'decoder.quantize').read_text())['quantize_parameters']
limits={}
for node,info in graph.items():
    q=quant.get('@'+node+':out0',{})
    name=info.get('name','')
    if name not in infos and '/'+name in infos:name='/'+name
    if q.get('qtype')=='i16' and name in infos:
        limits[name]={'fl':q['fl'],'limit':32767*2.**(-q['fl']),'clipped':0,'elements':0,'max_abs':0.,'worst_ratio':0.,'worst_position':0}
k=np.zeros((1,layer.kv,capacity,layer.hd),np.float32);v=k.copy()
with torch.inference_mode():
    for pos in range(len(tokens)):
        feed=feed_for(layer,hidden[:,pos:pos+1].numpy(),k,v,pos,capacity)
        values=dict(zip(names,session.run(names,feed)))
        for name,stats in limits.items():
            a=values[name];peak=float(np.abs(a).max());ratio=peak/stats['limit']
            stats['clipped']+=int(np.count_nonzero((a>stats['limit'])|(a < -32768*2.**(-stats['fl']))))
            stats['elements']+=a.size;stats['max_abs']=max(stats['max_abs'],peak)
            if ratio>stats['worst_ratio']:stats['worst_ratio']=ratio;stats['worst_position']=pos
        k[:,:,pos:pos+1]=values['new_k'];v[:,:,pos:pos+1]=values['new_v']
rows=sorted(({'name':key,**value} for key,value in limits.items()),key=lambda r:r['worst_ratio'],reverse=True)
report={'scope':'FP32 original-weight ONNX activation ranges versus compiled static INT16 limits; diagnosis, not NPU measurements',
        'layer':args.layer,'tokens':tokens,'ranges':rows}
(board/f'activation-ranges-l{args.layer}.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows[:20]:print(json.dumps(row),flush=True)
