from pathlib import Path
import json,hashlib,statistics,sys,numpy as np
r=Path(__file__).resolve().parent;f=Path('/opt/opi-lab/llama32-compat-20261023/fixtures');b=r/(sys.argv[1] if len(sys.argv)>1 else 'board-results')
rows=[json.loads(x) for x in (b/'probe.log').read_text().splitlines() if x.startswith('{')];metrics=[]
for v in rows:
 if v.get('event')!='fixture':continue
 pos=v['position'];x=np.fromfile(b/'outputs'/f'p{pos}-output.f32',dtype='<f4');y=np.fromfile(f/f'p{pos}-reference.f32',dtype='<f4')
 assert x.shape==y.shape==(2048,) and np.isfinite(x).all()
 rel=float(np.linalg.norm(x-y)/np.linalg.norm(y));cos=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
 assert abs(rel-v['hidden_relative_l2'])<1e-6
 metrics.append({**v,'independent_relative_l2':rel,'cosine':cos,'output_sha256':hashlib.sha256((b/'outputs'/f'p{pos}-output.f32').read_bytes()).hexdigest()})
summary=dict(scope=('One layer0 with 4 W8 FC and 3 FP16 MLP FC' if 'mlp-fp16' in b.name else 'One layer0 with all 7 FC W8')+' NPU execution; no full model or output head',status='EXECUTION_SUCCESS_PRECISION_REJECTED',fixtures=metrics,api_median_seconds=statistics.median(x['run_seconds'] for x in metrics),closed=rows[-1],precision_note='All six hidden relative L2 errors exceed historical5% probe screening threshold; these are six calibration states, not independent heldout cases; no full-model task accuracy implied')
(r/(sys.argv[2] if len(sys.argv)>2 else 'board-summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
