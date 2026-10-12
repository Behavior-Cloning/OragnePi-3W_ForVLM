"""Read our own compiler-generated weight bytes for an independent FP32 oracle."""
from pathlib import Path
import re
import numpy as np
import torch

def apply_exported_int8(layer, folder: Path):
    graph=(folder/'export/vnn_decoder.c').read_text()
    arrays={k:np.asarray([float(v.strip()) for v in body.split(',') if v.strip()],np.float32)
            for k,body in re.findall(r'float\s+scales_(\d+)\[\]\s*=\s*\{([^}]+)\}',graph)}
    raw=(folder/'export/decoder.export.data').read_bytes()
    data=np.load(folder/'decoder.data',allow_pickle=True).item()
    records=[]
    for name,parameters in data.items():
        if not name.startswith('fullconnect'):continue
        original=parameters['weight']
        block=graph.split('/* @'+name+':weight */',1)[1].split('/*',1)[0]
        match=re.search(r'VSI_NN_TYPE_(INT8|FLOAT16),\s*(\d+),\s*(\d+)\)',block)
        if not match:raise ValueError('expected INT8 or explicit mixed FP16 decoder matrix')
        dtype=match[1];offset,count=map(int,match.groups()[1:])
        itemsize=1 if dtype=='INT8' else 2
        quantized=np.frombuffer(raw,np.int8 if itemsize==1 else np.float16,count=count//itemsize,offset=offset).reshape(original.shape[::-1]).T
        scale_id=re.search(r'attr.dtype.scales = scales_(\d+)',block)
        if dtype=='FLOAT16':scale=1.
        elif scale_id:
            scale=arrays[scale_id[1]][None,:]
        else:
            fl=int(re.search(r'attr.dtype.fl = (-?\d+)',block)[1]);scale=2.**(-fl)
        restored=quantized.astype(np.float32)*scale
        candidates=[key for key in ('q','k','v','o','gate','up','down')
                    if getattr(layer,key).shape==original.T.shape and np.array_equal(getattr(layer,key).numpy(),original.T)]
        assert len(candidates)==1,(name,candidates)
        key=candidates[0]
        records.append({'matrix':key,'dtype':dtype,'relative_l2':float(np.linalg.norm(restored-original)/np.linalg.norm(original))})
        getattr(layer,key).copy_(torch.from_numpy(restored.T.copy()))
    assert len(records)==7
    return records
