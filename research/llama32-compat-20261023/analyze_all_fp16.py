"""Independently compare all-FP16 board outputs with the two saved variants."""
from pathlib import Path
import hashlib,json,statistics,numpy as np
r=Path(__file__).resolve().parent;fixtures=Path('/opt/opi-lab/llama32-compat-20261023/fixtures')
variants={'w8':'board-results','mlp_fp16':'board-results-mlp-fp16','all_fp16':'board-results-all-fp16'}
comparison={};summaries={}
for key,dirname in variants.items():
    d=r/dirname;rows=[json.loads(s) for s in (d/'probe.log').read_text().splitlines() if s.startswith('{')]
    events=[s for s in rows if s.get('event')=='fixture'];assert {s['position'] for s in events}=={0,1,2,4,8,14} and len(events)==6
    metrics=[]
    for event in events:
        p=event['position'];ref=np.fromfile(fixtures/f'p{p}-reference.f32',dtype='<f4').astype(np.float64);f=d/'outputs'/f'p{p}-output.f32';x=np.fromfile(f,dtype='<f4').astype(np.float64)
        assert x.shape==ref.shape==(2048,) and np.isfinite(x).all()
        rel=float(np.linalg.norm(x-ref)/np.linalg.norm(ref));assert abs(rel-event['hidden_relative_l2'])<1e-6
        item={**event,'independent_relative_l2':rel,'cosine':float(x@ref/(np.linalg.norm(x)*np.linalg.norm(ref))),'output_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        metrics.append(item);comparison.setdefault(p,{})[key]=item
    assert rows[-1]['event']=='closed' and rows[-1]['driver_released']==0
    summaries[key]={'api_median_seconds':statistics.median(x['run_seconds'] for x in metrics),'relative_l2_min':min(x['independent_relative_l2'] for x in metrics),'relative_l2_max':max(x['independent_relative_l2'] for x in metrics),'closed':rows[-1]}
d={'scope':'same six calibration states; actual one-layer NPU API times, not full-model TTFT/decode','summaries':summaries,'comparison':comparison,'screen_threshold_relative_l2':0.05,'all_fp16_passed_legacy_screen':all(x['all_fp16']['independent_relative_l2']<=.05 for x in comparison.values()),'note':'The 5% threshold is only a coarse compatibility screen, never full-model Chinese quality approval.'}
(r/'all-fp16-board-comparison.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
