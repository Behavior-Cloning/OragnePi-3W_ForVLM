from pathlib import Path
import json,numpy as np,statistics as st
P=Path(__file__).resolve().parent;s=json.loads((P/'selective-board-status.json').read_text());assert s['status']=='complete';rows=[]
for c in s['cases']:
 o=P/'selective-results/results';m=json.loads((o/(c['name']+'.json')).read_text());assert m['released_bytes']==0 and m['differing_elements']==0
 ref=np.fromfile(P/f"s224/l13-output{c['image']}.f32",np.float32).astype('float64');got=np.fromfile(o/(c['name']+'.f32'),np.float32).astype('float64');assert got.size==ref.size and np.isfinite(got).all()
 rows.append({**c,'relative_l2':float(np.linalg.norm(ref-got)/np.linalg.norm(ref)),'cosine':float(ref@got/(np.linalg.norm(ref)*np.linalg.norm(got))),'api_median_s':st.median(m['api_seconds']),'host_median_s':st.median(m['input_conversion_seconds'][i]+m['flush_seconds'][i]+m['readback_seconds'][i] for i in range(10)),'driver_released':0})
(P/'selective-audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
