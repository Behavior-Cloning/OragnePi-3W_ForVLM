#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,numpy as np,yaml
R=Path(__file__).resolve().parent
SRC=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def changes(a,b,path=''):
 if type(a)!=type(b):return [{'path':path,'before':a,'after':b}]
 if isinstance(a,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:out.append({'path':path+'/'+k,'before':a.get(k),'after':b.get(k)})
   else:out+=changes(a[k],b[k],path+'/'+k)
  return out
 if isinstance(a,list):
  if len(a)!=len(b):return [{'path':path,'before':a,'after':b}]
  out=[]
  for i,(x,y) in enumerate(zip(a,b)):out+=changes(x,y,path+'/'+str(i))
  return out
 return [] if a==b else [{'path':path,'before':a,'after':b}]
report={'scope':'layer0+1 only; graph/quantizer/weight audit, not board output or whole-model accuracy','cases':[]}
for name,variant in [('decode','w8-fused2-c1280-cap1-headmse'),('prefill16','w8-fused2-c1280-t16-cap1-productwide')]:
 old=SRC/variant/'l0';new=R/'artifacts'/name
 g0=json.loads((old/'decoder.json').read_text());g1=json.loads((new/'decoder.json').read_text())
 q0=yaml.safe_load((old/'decoder.quantize').read_text());q1=yaml.safe_load((new/'decoder.quantize').read_text())
 d0=np.load(old/'decoder.data',allow_pickle=True).item();d1=np.load(new/'decoder.data',allow_pickle=True).item()
 assert d0.keys()==d1.keys()
 wd=[];arrays=0;parameters=0
 for node in d0:
  assert d0[node].keys()==d1[node].keys()
  for key in d0[node]:
   a=np.asarray(d0[node][key]);b=np.asarray(d1[node][key]);arrays+=1;parameters+=a.size
   if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b,equal_nan=True):wd.append({'node':node,'parameter':key,'oldshape':a.shape,'newshape':b.shape})
 gd=changes(g0,g1);qd=changes(q0,q1)
 assert not wd,(name,wd)
 assert not qd,(name,qd)
 for x in gd:
  assert '/parameters/' in x['path'],x
  assert isinstance(x['before'],int) and isinstance(x['after'],int),x
  assert x['before']-x['after']==1024,x
 c=json.loads((new/'conversion.json').read_text());assert c['contract_verified_from_exported_c'] and c['capacity']==256
 report['cases'].append({'name':name,'graph_changes':gd,'quantizer_semantically_identical':not qd,'quantizer_bytes_identical':sha(old/'decoder.quantize')==sha(new/'decoder.quantize'),'weight_arrays_identical':not wd,'weight_arrays_compared':arrays,'weight_elements_compared':parameters,'nb_old_sha256':sha(old/'export_nbg_unify/network_binary.nb'),'nb_new_sha256':sha(new/'export_nbg_unify/network_binary.nb'),'artifacts':{str(p.relative_to(new)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [new/'decoder.json',new/'decoder.data',new/'decoder.quantize',new/'tensor-inputmeta.yml',new/'conversion.json',new/'export_nbg_unify/network_binary.nb']},'input_order':c['input_order'],'output_order':c['output_order']})
(R/'graph-quantization-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({x['name']:{k:x[k] for k in ['quantizer_semantically_identical','quantizer_bytes_identical','weight_arrays_compared','weight_elements_compared','nb_new_sha256']} for x in report['cases']},indent=2))
