from pathlib import Path
import json,numpy as np,torch,statistics as st
from model import Block
P=Path(__file__).resolve().parent;rows=[]
for c in json.loads((P/'isolate-cases.json').read_text()):
 n=c['name'];o=P/'isolate-results/results';m=json.loads((o/(n+'.json')).read_text());assert m['released_bytes']==0 and m['differing_elements']==0
 inp=np.fromfile(P/'isolate-assets'/c['input'],np.float32).reshape(1,257,1024)
 with torch.no_grad():ref=Block(c['layer']).eval()(torch.from_numpy(inp)).numpy().astype('float64').ravel()
 got=np.fromfile(o/(n+'.f32'),np.float32).astype('float64');assert np.isfinite(got).all()
 rows.append({**c,'relative_l2_same_input':float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),'cosine':float(ref@got/(np.linalg.norm(ref)*np.linalg.norm(got))),'api_median_s':st.median(m['api_seconds']),'host_median_s':st.median(m['input_conversion_seconds'][i]+m['flush_seconds'][i]+m['readback_seconds'][i] for i in range(10)),'driver_released':m['released_bytes']})
(P/'isolate-audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
