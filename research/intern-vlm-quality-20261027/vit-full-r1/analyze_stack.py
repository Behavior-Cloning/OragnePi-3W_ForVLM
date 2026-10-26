from pathlib import Path
import json,hashlib,statistics as st,numpy as np,torch,torch.nn.functional as F
from model import weights
P=Path(__file__).resolve().parent;O=P/'results';events=[json.loads(l) for l in (O/'events.jsonl').read_text().splitlines()]
assert events[0]['layers']==24 and events[-1]['event']=='closed' and events[-1]['driver_bytes']==0
status=json.loads((O/'status.json').read_text());assert status['returncode']==0 and not status['monitor_errors']
measures=[x for x in events if x['event']=='result' and x['run']>=0];assert len(measures)==6
rows=[]
for image in [1,2]:
 for layer in range(24):
  ref=np.fromfile(P/('s224/l%02d-output%d.f32'%(layer,image)),np.float32).astype('float64');f=O/('image%d-layer%02d.f32'%(image,layer));got=np.fromfile(f,np.float32).astype('float64')
  assert ref.size==got.size==257*1024 and np.isfinite(got).all()
  rows.append({'image':image,'layer':layer,'relative_l2':float(np.linalg.norm(ref-got)/np.linalg.norm(ref)),'cosine':float(ref@got/(np.linalg.norm(ref)*np.linalg.norm(got))),'max_abs':float(abs(ref-got).max()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 for run in [image-1,image+1,image+3]:
  a=np.fromfile(O/('run%d-output.f32'%run),np.float32);b=np.fromfile(O/('image%d-layer23.f32'%image),np.float32);assert np.array_equal(a,b)
 x=torch.from_numpy(np.fromfile(O/('image%d-layer23.f32'%image),np.float32).reshape(1,257,1024).copy())[:,:-1,:].contiguous()
 x=x.reshape(1,16,8,2048).permute(0,2,1,3).contiguous().reshape(1,8,8,4096).permute(0,2,1,3).contiguous().reshape(1,64,4096)
 with torch.no_grad():
  x=F.layer_norm(x,(4096,),weights['mm.model.mlp.0.weight'],weights['mm.model.mlp.0.bias'],1e-5)
  x=F.linear(x,weights['mm.model.mlp.1.weight'],weights['mm.model.mlp.1.bias']);x=F.gelu(x,approximate='tanh');x=F.linear(x,weights['mm.model.mlp.3.weight'],weights['mm.model.mlp.3.bias'])
 x.numpy().tofile(O/('projected-image%d.f32'%image))
audit={'scope':'24 resident ViT blocks only; precomputed stem, projector applied offline in WSL; not full on-board tower or VLM','layer_errors':rows,'stack_median_seconds':st.median(x['stack_seconds'] for x in measures),'api_sum_median_seconds':st.median(sum(x['api_seconds']) for x in measures),'host_copy_sync_median_seconds':st.median(x['host_copy_sync_seconds'] for x in measures),'ready':events[0],'closed':events[-1],'max_temperature_mC':max(x['temperature_mC'] for x in measures)}
(P/'stack-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print({k:v for k,v in audit.items() if k!='layer_errors'});print('last layer errors',[x for x in rows if x['layer']==23])
