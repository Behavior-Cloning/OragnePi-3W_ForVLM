from pathlib import Path
import numpy as np,json
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');rows=[]
for kind in ['linear0','linear0-stable','linear0-norm4d','linear0-epssafe','trace-linear0']:
 p=r/'results'/(kind+'-output.f32');tm=r/'results'/(kind+'-timing.json')
 if not p.exists() or not tm.exists():continue
 a=np.fromfile(p,np.float32);ref=np.fromfile(b/kind/'reference.f32',np.float32);counts=[('hidden',1024),('conv',24576),('state',262144)]
 if kind.startswith('trace'):counts=[(v['name'],v['count']) for v in json.loads((b/kind/'outputs.json').read_text())]
 out=[];offset=0
 for name,count in counts:
  x=a[offset:offset+count];y=ref[offset:offset+count];offset+=count;out.append({'name':name,'relative_l2':float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-30)),'max_abs':float(abs(x-y).max()),'finite':bool(np.isfinite(x).all())})
 assert offset==len(a)==len(ref)
 row={'kind':kind,'timing':json.loads(tm.read_text()),'errors':out};(r/'results'/(kind+'-probe.json')).write_text(json.dumps(row,indent=2));rows.append({'kind':kind,'seconds':row['timing']['mean_api_seconds'],'errors':out})
(r/'results/layer-probe-summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
