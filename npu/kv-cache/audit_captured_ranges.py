"""Find activation range loss on a captured real full-model graph call.

Uses original graph FP32 arithmetic on already quantized NPU input tensors.
This is a range diagnostic, not a hardware execution or quality acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from graph_reference import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(2)
    graph=json.loads((a.fixture/'decoder.json').read_text())
    conversion=json.loads((a.fixture/'conversion.json').read_text())
    quant=yaml.safe_load((a.fixture/'decoder.quantize').read_text())['quantize_parameters']
    data=np.load(a.fixture/'decoder.data',allow_pickle=True).item()
    raw=np.fromfile(a.fixture/'probe-input.f32',np.float32);offset=0;feeds={}
    for key in conversion['input_order']:
        node=graph['Layers'][key];shape=node['parameters']['shape'];count=int(np.prod(shape))
        feeds[node['name']]=torch.from_numpy(raw[offset:offset+count].reshape(shape).copy());offset+=count
    assert offset==raw.size
    with torch.inference_mode():
        _,values=evaluate(graph,data,feeds,return_intermediates=True)
    rows=[]
    for key,value in values.items():
        q=quant.get('@'+key+':out0',{})
        if q.get('quantizer')!='dynamic_fixed_point':continue
        dtype=q['qtype'];assert dtype in ('i8','i16','i32')
        bits=int(dtype[1:]);low=-(2.**(bits-1))*2.**(-q['fl']);high=(2.**(bits-1)-1)*2.**(-q['fl'])
        clipped=value.clamp(low,high);outside=(value<low)|(value>high)
        node=graph['Layers'][key]
        rows.append({'node':key,'name':node['name'],'op':node['op'],'qtype':dtype,'fl':q['fl'],
                     'range':[low,high],'observed_range':[float(value.min()),float(value.max())],
                     'count':value.numel(),'outside_count':int(outside.sum()),
                     'clipping_relative_l2':float((value-clipped).norm()/value.norm().clamp_min(1e-20))})
    rows.sort(key=lambda r:r['clipping_relative_l2'],reverse=True)
    report={'scope':__doc__,'fixture':str(a.fixture.resolve()),
            'input_sha256':hashlib.sha256((a.fixture/'probe-input.f32').read_bytes()).hexdigest(),
            'nodes':rows}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows[:20],indent=2))


if __name__=='__main__':main()
