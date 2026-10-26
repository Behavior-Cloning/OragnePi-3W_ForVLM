from pathlib import Path
import numpy as np,json,hashlib,statistics,re
P=Path(__file__).resolve().parent;j=json.loads((P/'results/summary.json').read_text());assert j['state']=='completed'
checks=[]
for i in [1,2]:
 a=np.fromfile(P/'results'/('off-before-%d.f32'%i),np.float32).astype('float64');b=np.fromfile(P/'results'/('repack-on-%d.f32'%i),np.float32).astype('float64');c=np.fromfile(P/'results'/('off-after-%d.f32'%i),np.float32)
 assert len(a)==len(b)==64*896 and np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a,c)
 checks.append({'image':i,'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'max_abs':float(abs(a-b).max()),'before_after_equal':True})
assert 'OPI Q8 CPU_REPACK tensors=144' in (P/'results/repack-on.stderr').read_text()
phase=[]
for x in j['phases']:
 enc=[e for e in x['events'] if e.get('event')=='encoded'];assert len(enc)==3 and x['returncode']==0 and x['events'][-1]['event']=='closed'
 phase.append({'name':x['phase'],'hot_median_ms':statistics.median(e['encoder_projector_ms'] for e in enc[1:]),'load_ms':x['events'][0]['load_ms']})
a={'scope':'224->64 CPU vision Q8 identical binary/model off-on-off; 2 hot samples per phase','phases':phase,'embedding_checks':checks,'verified_repacked_tensor_count':144}
(P/'AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
