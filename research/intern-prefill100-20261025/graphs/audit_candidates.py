#!/usr/bin/env python3
"""Independent post-build graph/precision/port audit; never touches board."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np,yaml
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def walk(a,b,path=''):
 assert type(a)==type(b),path
 if isinstance(a,dict):
  assert set(a)==set(b),path
  return sum([walk(a[k],b[k],path+'/'+k) for k in a],[])
 if isinstance(a,list):
  assert len(a)==len(b),path
  return sum([walk(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [{'path':path,'before':a,'after':b}]
s=json.loads((R/'status.json').read_text());assert s['state']=='complete',s['state']
prov=json.loads((R/'source-provenance.json').read_text());rows=[]
for row in s['cases']:
 name=row['id'];p=R/'cases'/name;first=row['first_layer'];m=row['query_tokens'];c=row['capacity'];src=Path(prov['sources'][str(first)]['path']);source=prov['sources'][str(first)]
 for file,h in source['hashes'].items():assert sha(src/file)==h,(name,file,'source drift')
 assert sha(p/'export_nbg_unify/network_binary.nb')==row['nb_sha256']
 for file in ['decoder.data','decoder.quantize','export/decoder.export.data']:
  assert sha(p/file)==source['hashes'][file],(name,file)
 assert (p/'decoder.data').is_symlink() and (p/'decoder.data').resolve()==(src/'decoder.data').resolve()
 assert (p/'export/decoder.export.data').is_symlink() and (p/'export/decoder.export.data').resolve()==(src/'export/decoder.export.data').resolve()
 old=json.loads((src/'decoder.json').read_text());new=json.loads((p/'decoder.json').read_text());changes=walk(old,new);assert changes==row['graph_changes']
 assert len(changes)==(10 if m==16 else 49)
 for change in changes:
  assert '/parameters/' in change['path']
  before,after=change['before'],change['after'];assert isinstance(before,int) and isinstance(after,int)
  assert (before,after) in [(1280,c),(1296,c+m),(16,m),(112,7*m)]
 met=json.loads((p/'export_nbg_unify/nbg_meta.json').read_text());oldmet=json.loads((src/'export_nbg_unify/nbg_meta.json').read_text())
 assert set(met)==set(oldmet)
 for kind in ['Inputs','Outputs']:
  assert set(met[kind])==set(oldmet[kind])
  for key,v in met[kind].items():
   a,b=copy.deepcopy(v),copy.deepcopy(oldmet[kind][key]);a.pop('shape');b.pop('shape');assert a==b,(name,kind,key,'port precision changed')
 assert len(met['Inputs'])==11 and len(met['Outputs'])==5
 for key,v in met['Inputs'].items():
  if 'past_' in key:assert v['shape']==[1,2,c,64]
  elif 'mask_' in key:assert v['shape']==[1,1,7*m,c+m]
  elif 'rope_' in key:assert v['shape']==[1,1,m,64]
  else:assert v['shape']==[1,m,896]
 for key,v in met['Outputs'].items():assert v['shape']==([1,m,896] if 'hidden' in key else [1,2,m,64])
 arrays=np.load(p/'decoder.data',allow_pickle=True).item();count=0
 for spec in source['arrays']:
  v=np.asarray(arrays[spec['node']][spec['field']]);assert list(v.shape)==spec['shape'] and str(v.dtype)==spec['dtype'] and hashlib.sha256(v.tobytes()).hexdigest()==spec['sha256'];count+=1
 rows.append({'id':name,'nb_sha256':row['nb_sha256'],'nb_bytes':row['nb_bytes'],'source_weight_arrays_rehashed':count,'changed_integer_fields':len(changes),'quantizer_file_byte_equal':True,'exported_quantized_payload_byte_equal':True,'compiled_port_quantization_identical':True,'input_output_shape_contract_pass':True,'board_numerical_validation':'pending'})
out={'passed':True,'scope':'graph/weights/port metadata only; no measured speed or end-to-end accuracy implied','audited_unix':time.time(),'rows':rows,'total_nb_bytes':sum(x['nb_bytes'] for x in rows),'space':s['space_end'],'compiler_seconds_sum':sum(x['compile_seconds'] for x in s['cases'])}
(R/'independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
