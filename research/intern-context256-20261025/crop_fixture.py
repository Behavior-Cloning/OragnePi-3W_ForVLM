#!/usr/bin/env python3
"""Crop only masked history padding from a frozen same-query fixture.
Accepts FP32 packed tensor_probe input; never regenerates effective KV or RoPE.
"""
from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--history',type=int,required=True);a=p.parse_args()
def sha(b):return hashlib.sha256(b).hexdigest()
c0=json.loads((a.old/'conversion.json').read_text());c1=json.loads((a.new/'conversion.json').read_text())
g0=json.loads((a.old/'decoder.json').read_text())['Layers'];g1=json.loads((a.new/'decoder.json').read_text())['Layers']
C0,C1=c0['capacity'],c1['capacity'];T=c0['query_tokens']
assert C1<C0 and 0<=a.history and a.history+T<=C1
assert T==c1['query_tokens'] and c0['key_cache_transposed']==c1['key_cache_transposed']==False
assert c0['input_order']==c1['input_order'] and c0['output_order']==c1['output_order']
raw=a.input.read_bytes();flat=np.frombuffer(raw,dtype='<f4');offset=0;chunks=[];rows=[]
for k in c0['input_order']:
 name=g0[k]['name'];shape0=g0[k]['parameters']['shape'];shape1=g1[k]['parameters']['shape'];n=int(np.prod(shape0));x=flat[offset:offset+n].reshape(shape0);offset+=n
 assert np.isfinite(x).all(),name
 if name.endswith(('_past_k','_past_v')):
  y=x[:,:,:C1,:].copy()
  assert np.array_equal(y[:,:,:a.history,:],x[:,:,:a.history,:])
 elif name.endswith('_mask'):
  assert x.shape[-1]==C0+T
  assert (x[...,a.history:C0]<=-1000).all(),'unmasked past entries exceed recorded history'
  assert (x[...,C1:C0]<=-1000).all(),'cannot remove effective keys'
  y=np.concatenate((x[...,:C1],x[...,C0:]),axis=-1)
 else:y=x.copy()
 assert list(y.shape)==shape1,(name,y.shape,shape1)
 chunks.append(y.ravel());rows.append({'name':name,'old_shape':shape0,'new_shape':shape1,'old_sha256':sha(x.tobytes()),'new_sha256':sha(y.tobytes()),'effective_values_retained_exactly':True})
assert offset==len(flat)
a.output.mkdir(parents=True,exist_ok=False)
packed=np.concatenate(chunks).astype('<f4').tobytes();(a.output/'probe-input.f32').write_bytes(packed)
report={'old_graph':str(a.old),'new_graph':str(a.new),'input_source':str(a.input),'old_input_sha256':sha(raw),'new_input_sha256':sha(packed),'old_capacity':C0,'new_capacity':C1,'query_tokens':T,'history':a.history,'source_history_kind':'independent FP32 model history when using prepared fixture, not captured NPU KV; same source feeds cropped without recalculation','inputs':rows,'output_comparison':'only current-query hidden and new K/V are output; identical query count means compare all output elements; no history outputs/padding'}
(a.output/'crop-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'output':str(a.output),'bytes':len(packed),'old_bytes':len(raw),'history':a.history,'query_tokens':T}))
