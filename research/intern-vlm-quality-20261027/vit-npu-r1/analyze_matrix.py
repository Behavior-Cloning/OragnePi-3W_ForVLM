from pathlib import Path
import json,hashlib,statistics as st,numpy as np
P=Path(__file__).resolve().parent;j=json.loads((P/'matrix-results/summary.json').read_text());rows=[]
assert j['status']=='complete' and len(j['records'])==14
for x in j['records']:
 f=P/'matrix-results'/(x['name']+'.f32');assert hashlib.sha256(f.read_bytes()).hexdigest()==x['output_sha256']
 a=np.fromfile(P/('s%d/output%d.f32'%(x['side'],x['image'])),np.float32).astype('float64');b=np.fromfile(f,np.float32).astype('float64');assert len(a)==len(b) and np.isfinite(b).all()
 m=x['metrics'];assert m['released_bytes']==0 and m['differing_elements']==0
 rows.append({**{k:x[k] for k in ['name','side','dtype','image']},'api_median_ms':1000*st.median(m['api_seconds']),'input_conversion_ms':1000*st.median(m['input_conversion_seconds']),'flush_ms':1000*st.median(m['flush_seconds']),'readback_ms':1000*st.median(m['readback_seconds']),'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'max_abs':float(abs(a-b).max()),'driver_bytes':m['driver_bytes']})
(P/'matrix-audit.json').write_text(json.dumps({'scope':'ViT layer0 only; two development images; no full-tower extrapolation','records':rows},indent=2)+'\n')
for x in rows:print(x['name'],round(x['api_median_ms'],3),round(100*x['relative_l2'],3),round(x['input_conversion_ms']+x['flush_ms']+x['readback_ms'],3))
