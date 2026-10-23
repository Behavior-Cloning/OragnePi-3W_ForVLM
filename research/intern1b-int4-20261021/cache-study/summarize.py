from pathlib import Path
import json,statistics,hashlib
import numpy as np
r=Path(__file__).resolve().parent;p=sorted((r/'results').glob('*/events.jsonl'))[-1];rows=[json.loads(x) for x in p.read_text().splitlines()]
summary={'complete':any(x['event']=='complete' for x in rows),'errors':[x for x in rows if x['event']=='error'],'cases':{},'result_file':str(p),'scope':'W8 tail_min8; both model-resident; only system KV reuse varied; actual text inputs; no images'}
def stats(a):return {'n':len(a),'p50':statistics.median(a),'min':min(a),'max':max(a)}
for case in json.loads((r/'inputs.json').read_text())['cases']:
 out={'system_tokens':len(case['prefix']),'dynamic_tokens':case['dynamic_tokens']}
 for cache in [False,True]:
  group=[x for x in rows if x['event']=='measure' and x['case']==case['id'] and x['cache']==cache];assert len(group)==3
  assert all(x['result']['reused_tokens']==(len(case['prefix']) if cache else 0) for x in group)
  out['cached' if cache else 'recompute']={k:stats([x['result'][k] for x in group]) for k in ['ttft_seconds','prefill_seconds','setup_seconds','scalar_prompt_seconds','first_head_seconds','cpu_seconds']}
  out['cached' if cache else 'recompute']['prepare_seconds']=stats([x['prepare_seconds'] for x in group])
 diag=[x for x in rows if x['event']=='diagnostic' and x['case']==case['id']];assert len(diag)==2
 a,b=[np.fromfile(p.parent/'logits'/Path(x['dump']).name,np.float32) for x in diag];assert a.shape==b.shape==(151674,)
 out['first_token_logits']={'bit_identical':bool(np.array_equal(a,b)),'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'argmax_equal':bool(a.argmax()==b.argmax())}
 cold=out['recompute']['ttft_seconds']['p50'];warm=out['cached']['ttft_seconds']['p50'];out['saved_seconds']=cold-warm;out['reduction_percent']=100*(1-warm/cold);summary['cases'][case['id']]=out
snapshots=[x['resources'] for x in rows if 'resources' in x];summary['resources']={'temperature_max':max(x['temperature'] for x in snapshots),'available_min':min(x['available'] for x in snapshots),'swap_in_delta':max(x['swap_in'] for x in snapshots)-min(x['swap_in'] for x in snapshots),'swap_out_delta':max(x['swap_out'] for x in snapshots)-min(x['swap_out'] for x in snapshots),'note':'request boundary snapshots, not continuous peak sampling'}
summary['closed']=next(x['result'] for x in rows if x['event']=='closed');assert summary['complete'] and not summary['errors'] and summary['closed']['driver_bytes']==0
(r/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
