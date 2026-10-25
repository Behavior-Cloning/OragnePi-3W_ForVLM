#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys,numpy as np,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]/'npu/kv-cache'))
from graph_reference import evaluate
torch.set_num_threads(2)
def execute(folder,inp):
 c=json.loads((folder/'conversion.json').read_text());g=json.loads((folder/'decoder.json').read_text());data=np.load(folder/'decoder.data',allow_pickle=True).item();x=np.fromfile(inp,dtype='<f4');feeds={};off=0
 for key in c['input_order']:
  node=g['Layers'][key];shape=node['parameters']['shape'];n=int(np.prod(shape));feeds[node['name']]=torch.from_numpy(x[off:off+n].reshape(shape).copy());off+=n
 assert off==len(x)
 with torch.inference_mode():return {k:v.numpy().copy() for k,v in evaluate(g,data,feeds).items()}
report={'scope':'FP32 graph capacity equivalence on identical real-model history; not NPU precision or full-model validation','cases':[]}
for name in ['decode','prefill16']:
 old=R/'fixtures'/('cap1280-'+name);new=R/'artifacts'/name
 a=execute(old,old/'probe-input.f32');b=execute(new,R/'fixtures'/('cap256-'+name)/'probe-input.f32');assert a.keys()==b.keys();row={'name':name,'outputs':{}}
 for k,x in a.items():
  y=b[k];e=(y.astype('float64')-x);rel=float(np.linalg.norm(e)/max(np.linalg.norm(x.astype('float64')),1e-30));maximum=float(abs(e).max())
  assert np.isfinite(y).all() and rel<1e-5 and maximum<1e-3,(name,k,rel,maximum)
  row['outputs'][k]={'elements':x.size,'bitwise_equal':x.tobytes()==y.tobytes(),'relative_l2':rel,'max_abs':maximum,'old_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'new_sha256':hashlib.sha256(y.tobytes()).hexdigest()}
 report['cases'].append(row)
(R/'fixture-graph-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
