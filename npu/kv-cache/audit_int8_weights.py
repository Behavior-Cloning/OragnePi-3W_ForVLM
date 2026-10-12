"""Check exported INT8 weight bytes/scales and a host-only arithmetic oracle."""
import argparse
import json
from pathlib import Path
import re
import gguf
import numpy as np
import torch
from trained_probe import TrainedDecoder

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--fixture',type=Path,required=True);p.add_argument('--layer0',type=Path,required=True)
p.add_argument('--layer1',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();torch.set_num_threads(2);reader=gguf.GGUFReader(args.model)
manifest=json.loads((args.fixture/'trained-manifest.json').read_text())
x=torch.from_numpy(np.fromfile(args.fixture/'hidden.f32',np.float32).reshape(1,-1,manifest['dim']))
reference=np.fromfile(args.fixture/'reference.f32',np.float32).reshape(-1,manifest['dim'])
records=[]
with torch.inference_mode():
    for index,folder in enumerate((args.layer0,args.layer1)):
        graph=(folder/'export/vnn_decoder.c').read_text()
        arrays={k:np.asarray([float(v.strip()) for v in body.split(',') if v.strip()],np.float32)
                for k,body in re.findall(r'float\s+scales_(\d+)\[\]\s*=\s*\{([^}]+)\}',graph)}
        raw=(folder/'export/decoder.export.data').read_bytes()
        data=np.load(folder/'decoder.data',allow_pickle=True).item()
        layer=TrainedDecoder(reader,index).eval();matched=set()
        for name,parameters in data.items():
            if not name.startswith('fullconnect'):continue
            original=parameters['weight']
            block=graph.split('/* @'+name+':weight */',1)[1].split('/*',1)[0]
            scale_id=re.search(r'attr.dtype.scales = scales_(\d+)',block)[1]
            match=re.search(r'VSI_NN_TYPE_INT8,\s*(\d+),\s*(\d+)\)',block)
            offset,count=map(int,match.groups())
            quantized=np.frombuffer(raw,np.int8,count=count,offset=offset).reshape(original.shape[::-1]).T
            restored=quantized.astype(np.float32)*arrays[scale_id][None,:]
            candidates=[key for key in ('q','k','v','o','gate','up','down')
                        if getattr(layer,key).shape==original.T.shape and np.array_equal(getattr(layer,key).numpy(),original.T)]
            assert len(candidates)==1,(name,candidates)
            key=candidates[0];matched.add(key)
            records.append({'layer':index,'matrix':key,'exported_weight_name':name,
                'relative_l2':float(np.linalg.norm(restored-original)/np.linalg.norm(original)),
                'max_abs_error':float(np.max(np.abs(restored-original))),
                'scale_channels':len(arrays[scale_id]),'output_channels':original.shape[1]})
            getattr(layer,key).copy_(torch.from_numpy(restored.T.copy()))
        assert len(matched)==7
        x=layer.full(x)
    actual=x.numpy()[0]
cos=(actual*reference).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(reference,axis=-1))
report={'scope':'exported INT8 weight bytes reconstructed and evaluated with FP32 activations on host; not board speed',
        'weights':records,'two_layer_relative_l2':float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
        'two_layer_min_cosine':float(cos.min())}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
