from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;D=P/'results/suffixgroup-firstfix-r1'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert json.loads((P/'controller.json').read_text())['returncode']==0
assert json.loads((D/'meta.json').read_text())['status']=='complete'
events=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];rows=[x for x in events if x['event']=='result'];closed=[x for x in events if x['event']=='closed'];assert len(rows)==24 and len(closed)==3 and all(x['metrics']['driver_bytes']==0 for x in closed)
for row in rows:
 f=D/row['logits_file'];assert sha(f)==row['logits_sha256'];a=np.fromfile(f,dtype='<f4');assert a.size==row['metrics']['generated_tokens']*151674 and np.isfinite(a).all()
comparisons=[]
for case in sorted({r['case'] for r in rows}):
 for kind in ['fixed','natural']:
  rr={r['variant']:r for r in rows if r['case']==case and r['kind']==kind};base=rr['original-grouped'];aa=np.fromfile(D/base['logits_file'],dtype='<f4').reshape(-1,151674)
  c=dict(case=case,kind=kind,variants={})
  for name,r in rr.items():
   b=np.fromfile(D/r['logits_file'],dtype='<f4').reshape(-1,151674);n=min(len(aa),len(b));delta=b[:n].astype(float)-aa[:n]
   c['variants'][name]=dict(reply=r['reply'],tokens=r['metrics']['tokens'],output_tokens=r['metrics']['generated_tokens'],tokens_exact=r['metrics']['tokens']==base['metrics']['tokens'],logits_sha_exact=r['logits_sha256']==base['logits_sha256'],relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(aa[:n].astype(float)),1e-30)),top1_equal=int((aa[:n].argmax(1)==b[:n].argmax(1)).sum()),steps=n,numerical_comparison_valid_same_feedback=(kind=='fixed'),note='Natural autoregressive logit differences after divergent output are not same-input errors' if kind=='natural' else 'Same fixed feedback, full vocabulary')
  comparisons.append(c)
a=dict(status='complete',scope=json.loads((D/'config.json').read_text())['scope'],counts=dict(results=24,closed_zero=3),comparisons=comparisons,first_chunk_switches={x['variant']:x['original_first_chunk_calls'] for x in closed})
a['hybrid_group_parity']=[]
for case in sorted({r['case'] for r in rows}):
 for kind in ['fixed','natural']:
  rr={r['variant']:r for r in rows if r['case']==case and r['kind']==kind};ok=rr['hybrid-suffixgroup']['logits_sha256']==rr['hybrid-ungrouped']['logits_sha256'];a['hybrid_group_parity'].append(dict(case=case,kind=kind,exact=ok));assert ok,'suffix grouping changed logits'
(P/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
for c in comparisons:
 print(c['case'],c['kind'])
 for n,v in c['variants'].items():print(n,v['reply'],round(v['relative_l2'],5),v['top1_equal'],v['steps'])
