from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;D=R/'results';e=[json.loads(l) for l in (D/'events.jsonl').read_text().splitlines()]
rows=[];base=None
for x in e:
 if not x.get('diagnostic'):continue
 p=D/x['logits'];b=p.read_bytes();a=np.frombuffer(b,dtype='<f4');assert len(b)==2426784 and hashlib.sha256(b).hexdigest()==x['sha256'] and np.isfinite(a).all()
 if base is None:base=a.copy()
 rows.append({'phase':x['phase'],'repeat':x['repeat'],'sha256':x['sha256'],'byte_equal_original':np.array_equal(a,base),'relative_l2':float(np.linalg.norm(a-base)/np.linalg.norm(base)),'argmax':a.reshape(4,151674).argmax(axis=1).tolist()})
assert len(rows)==12
assert all(x['byte_equal_original'] for x in rows if x['phase']!='range16')
closed=[x for x in e if 'closed' in x];assert len(closed)==4 and all(x['closed']['driver_bytes']==0 for x in closed)
quality={}
for phase in ['original','original-tail8','original-repeat','range16']:
 q=[x for x in e if x.get('quality') and x['phase']==phase];quality[phase]={x['case']:x['events'][-1]['tokens'] for x in q}
parity={k:sum(v[n]==quality['original'][n] for n in v) for k,v in quality.items()}
a={'status':'complete','controller_returncode':json.loads((R/'controller.json').read_text())['returncode'],'full_logits_diagnostics':rows,'natural_token_parity_out_of_6':parity,'closed_driver0':4,'performance_claim':None,'reason':'diagnostic=True; timings excluded from speed claims; no concurrent voice/RAG in r3','goal_met':False}
(R/'ANALYSIS.json').write_text(json.dumps(a,indent=2))
print(json.dumps({'diagnostic_count':len(rows),'parity':parity,'range16_l2':[x['relative_l2'] for x in rows if x['phase']=='range16']}))
