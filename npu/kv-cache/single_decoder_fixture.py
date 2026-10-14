"""Isolate the first decoder on actual prompt tokens, including K/V outputs.

This is a diagnostic fixture, never a full-model acceptance result.
"""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from exported_int8 import apply_exported_int8
from trained_probe import TrainedDecoder, feed_for
from probe import INPUTS, OUTPUTS

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--folder',type=Path,required=True)
p.add_argument('--prompt',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--position',type=int,default=0)
p.add_argument('--capacity',type=int,default=128)
p.add_argument('--layer',type=int,default=0)
p.add_argument('--prefix-stack',type=Path,help='reconstruct preceding layers with their exported weights')
p.add_argument('--capture',type=Path,help='actual ungrouped full_model decode input; overrides constructed inputs')
a=p.parse_args();torch.set_num_threads(2)
ids=json.loads(a.prompt.read_text())['token_ids'][:a.position+1]
assert len(ids)==a.position+1 and 0<=a.position<a.capacity
a.output.mkdir(parents=True,exist_ok=False)
for name in ('export_nbg_unify',):
    (a.output/name).symlink_to((a.folder/name).resolve(),target_is_directory=True)
(a.output/'conversion.json').write_text(json.dumps({'fusion_width':1}))
r=gguf.GGUFReader(a.model)
t=next(t for t in r.tensors if t.name=='token_embd.weight')
x=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type)[ids][None].copy())
layer=TrainedDecoder(r,a.layer,True,strict_mask=True).eval()
with torch.inference_mode():
    for i in range(a.layer):
        previous=TrainedDecoder(r,i,True).eval()
        if a.prefix_stack:apply_exported_int8(previous,a.prefix_stack/f'l{i}')
        x=previous.full(x)
    constants=apply_exported_int8(layer,a.folder)
    _,k,v=layer.project(x,*layer.rope(torch.arange(len(ids))))
    pk=torch.zeros(1,layer.kv,a.capacity,layer.hd);pv=pk.clone()
    pk[:,:,:a.position]=k[:,:,:a.position];pv[:,:,:a.position]=v[:,:,:a.position]
    feeds=feed_for(layer,x[:,-1:].numpy(),pk.numpy(),pv.numpy(),a.position,a.capacity)
    if a.capture:
        raw=np.fromfile(a.capture,np.float32);offset=0
        for name in INPUTS:
            shape=feeds[name].shape;count=feeds[name].size
            feeds[name]=raw[offset:offset+count].reshape(shape).copy();offset+=count
        assert offset==raw.size and np.isfinite(raw).all()
    result=layer(*(torch.from_numpy(feeds[n]) for n in INPUTS))
np.concatenate([feeds[n].ravel() for n in INPUTS]).tofile(a.output/'probe-input.f32')
np.concatenate([t.numpy().ravel() for t in result]).tofile(a.output/'probe-reference.f32')
(a.output/'probe-fixture.json').write_text(json.dumps({
    'scope':'isolated layer, exported weights and independent FP32 arithmetic; not captured board history',
    'layer':a.layer,'prefix_stack':str(a.prefix_stack) if a.prefix_stack else None,
    'capture':str(a.capture) if a.capture else None,
    'position':a.position,'token_ids':ids,'constants':constants,
    'outputs':[(n,t.numel()) for n,t in zip(OUTPUTS,result)]},indent=2))
