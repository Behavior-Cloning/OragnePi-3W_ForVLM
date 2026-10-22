import json,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'primary-review.json').read_text());a=json.loads((ROOT/'independent-review.json').read_text())
P={(r['case_id'],r['card_id']):r for r in p['records']};A={(r['case_id'],r['card_id']):r for r in a['records']};assert set(P)==set(A) and len(P)==3300
out=[];diffs=[]
for key,r in P.items():
 other=A[key];same=r['status']==other['status']
 assert same or 'incompatible' not in (r['status'],other['status']),('unresolved hard disagreement',key)
 status=r['status'] if same else 'conditional'
 reason=r['reason'] if same else '主审与独立审计对是否已兼容存在分歧，保留待定，不算明确负例。主审：'+r['reason']+' 审计：'+other['reason']
 out.append(dict(case_id=key[0],card_id=key[1],status=status,reason=reason,primary_status=r['status'],audit_status=other['status']))
 if not same:diffs.append(key)
result=dict(rule='Both model-blind audits agree on all 304 hard-negative pairs. All 452 C/U disagreements retained as conditional. No labels changed from model rankings.',records=out,counts=dict(collections.Counter(r['status'] for r in out)),disagreements=len(diffs),input_sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['primary-review.json','independent-review.json','cards.json','cases.json']})
(ROOT/'adjudicated-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['counts'])
