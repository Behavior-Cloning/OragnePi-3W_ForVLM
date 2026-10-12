"""Export real Q projection alone, with independent calibration and test inputs."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import onnx
import torch
from torch import nn
from trained_probe import TrainedDecoder

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--matrix',choices=['q','v'],default='q')
args=p.parse_args();torch.set_num_threads(1)
out=args.output;out.mkdir(exist_ok=True,parents=True)
reader=gguf.GGUFReader(args.model);layer=TrainedDecoder(reader,0).eval()
weight=getattr(layer,args.matrix)
linear=nn.Linear(layer.dim,weight.shape[0],bias=False).eval();linear.weight.data.copy_(weight)
hidden=np.fromfile(args.fixture/'hidden.f32',np.float32).reshape(-1,layer.dim)
with torch.inference_mode():
    inputs=layer.norm(torch.from_numpy(hidden[:32,None,:].copy()),layer.norm1).numpy()
    extra=np.random.default_rng(42).normal(0,0.1,(3,1,layer.dim)).astype(np.float32)
    extra[0]*=0;extra[1]*=0.01
    inputs=np.concatenate([extra,inputs],axis=0)
    outputs=linear(torch.from_numpy(inputs)).numpy()
    inputs.tofile(out/'linear-input.f32');outputs.tofile(out/'linear-reference.f32')
    torch.onnx.export(linear,torch.ones(1,1,layer.dim),str(out/'layer0.onnx'),
                      input_names=['hidden'],output_names=['output'],opset_version=13,dynamo=False)
    model=onnx.load(out/'layer0.onnx');model.ir_version=8;onnx.save(model,out/'layer0.onnx')
    entries=[]
    for i,path in enumerate(sorted(args.fixture.glob('l0_p*_hidden.npy'))):
        h=np.load(path)
        x=layer.norm(torch.from_numpy(h),layer.norm1).numpy()
        name=f'calibration-{i}.npy';np.save(out/name,x);entries.append(name)
    (out/'layer0-dataset.txt').write_text('\n'.join(entries)+'\n')
(out/'fixture.json').write_text(json.dumps({'input_count':layer.dim,'output_count':int(weight.shape[0]),'cases':len(inputs),
    'scope':f'layer 0 {args.matrix} projection, input RMSNorm on FP32 host, no attention or KV'},indent=2)+'\n')
