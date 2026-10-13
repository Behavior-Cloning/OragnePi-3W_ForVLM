"""Find activation range loss on a captured real full-model graph call.

Uses graph FP32 arithmetic on already quantized NPU input tensors, with the
original or explicitly selected exported weights. No intermediate quantization
simulation is assumed; exported mode is checked against an independent oracle.
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
    p.add_argument('--exported-model',type=Path,
                   help='audit activations using actual exported weights, checked against the independent fixture')
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(2)
    graph=json.loads((a.fixture/'decoder.json').read_text())
    conversion=json.loads((a.fixture/'conversion.json').read_text())
    quant=yaml.safe_load((a.fixture/'decoder.quantize').read_text())['quantize_parameters']
    data=np.load(a.fixture/'decoder.data',allow_pickle=True).item()
    if a.exported_model:
        from exported_graph_data import restore
        # Fixtures link graph files to the exact compiler output directory.
        folder=(a.fixture/'decoder.data').resolve().parent
        data=restore(graph,data,folder,conversion,a.exported_model)
    raw=np.fromfile(a.fixture/'probe-input.f32',np.float32);offset=0;feeds={}
    for key in conversion['input_order']:
        node=graph['Layers'][key];shape=node['parameters']['shape'];count=int(np.prod(shape))
        feeds[node['name']]=torch.from_numpy(raw[offset:offset+count].reshape(shape).copy());offset+=count
    assert offset==raw.size
    with torch.inference_mode():
        outputs,values=evaluate(graph,data,feeds,return_intermediates=True)
        if a.exported_model:
            actual=np.concatenate([outputs[graph['Layers'][k]['name']].numpy().ravel()
                                   for k in conversion['output_order']])
            expected=np.fromfile(a.fixture/'probe-reference.f32',np.float32)
            np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=1e-4)
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
    report={'scope':__doc__,'reference_weights':'exported' if a.exported_model else 'original',
            'independent_output_checked':bool(a.exported_model),'fixture':str(a.fixture.resolve()),
            'input_sha256':hashlib.sha256((a.fixture/'probe-input.f32').read_bytes()).hexdigest(),
            'nodes':rows}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows[:20],indent=2))


if __name__=='__main__':main()
