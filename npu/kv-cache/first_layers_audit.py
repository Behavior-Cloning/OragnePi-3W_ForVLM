"""Build a genuine two-layer reference and audit first-step hidden/K/V outputs."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from trained_probe import TrainedDecoder, feed_for
from exported_int8 import apply_exported_int8

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--dtype');p.add_argument('--tag');args=p.parse_args();torch.set_num_threads(2)
manifest=json.loads((args.source/'trained-manifest.json').read_text())
reader=gguf.GGUFReader(args.model);args.output.mkdir(parents=True,exist_ok=True)
hidden=np.fromfile(args.source/'hidden.f32',np.float32).reshape(1,-1,manifest['dim'])
with torch.inference_mode():
    layers=[TrainedDecoder(reader,i).eval() for i in range(2)]
    if args.tag is None:
        result=torch.from_numpy(hidden.copy())
        for layer in layers:result=layer.full(result)
        hidden.tofile(args.output/'hidden.f32');result.numpy().tofile(args.output/'reference.f32')
        (args.output/'fixture.json').write_text(json.dumps({'scope':'first two layers only, original Q8 GGUF weights',
            'model':str(args.model),'source':str(args.source),'manifest':manifest},indent=2)+'\n')
    else:
        if not args.dtype:raise ValueError('dtype required for trace audit')
        x=hidden[:,:1].copy();rows=[]
        for i,layer in enumerate(layers):
            constants=apply_exported_int8(layer,args.source/args.dtype/f'l{i}')
            cache=np.zeros((1,layer.kv,manifest['capacity'],layer.hd),np.float32)
            feeds=feed_for(layer,x,cache,cache,0,manifest['capacity'])
            expected=layer(*(torch.from_numpy(v) for v in feeds.values()))
            raw=np.fromfile(args.output/args.tag/f'trace-layer{i}.f32',np.float32)
            sizes=[v.numel() for v in expected];actual=np.split(raw,np.cumsum(sizes)[:-1])
            for name,reference,value in zip(['hidden','new_k','new_v'],expected,actual):
                ref=reference.numpy().reshape(-1)
                rows.append({'layer':i,'output':name,'relative_l2':float(np.linalg.norm(value-ref)/max(np.linalg.norm(ref),1e-20)),
                    'max_abs_error':float(np.abs(value-ref).max()),'npu_nonzero':int(np.count_nonzero(value)),
                    'reference_nonzero':int(np.count_nonzero(ref)),'npu_min':float(value.min()),'npu_max':float(value.max()),
                    'reference_min':float(ref.min()),'reference_max':float(ref.max())})
            x=actual[0].reshape(1,1,layer.dim).copy()
        report={'scope':'first step, isolated local error at each layer using actual incoming NPU hidden; FP32 exported decoder constants', 'outputs':rows}
        (args.output/args.tag/'first-step-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
