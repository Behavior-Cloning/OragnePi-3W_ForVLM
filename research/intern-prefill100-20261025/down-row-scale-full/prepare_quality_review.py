"""Reuse manual judgments only for the identical frozen prompt+gold+reply; leave changes pending."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P.parent;D=P/'results/down-row-scale-full-r1'
J=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=R/'capacity512-full/results/c512-m16-m32-aba-r1';review=J(old/'ROOT_QUALITY_REVIEW.json');a=J(D/'analysis.json');f=J(D/'fixtures.json');of=J(old/'fixtures.json')
assert f['quality']==of['quality'],'Do not reuse judgments on changed inputs'
prior={r['case']:r for r in review['rows']};rows=[]
for q in a['natural_quality']:
 case=q['id'];r=prior[case];assert q['gold']['required_action']==r['required'] and q['gold']['forbidden_actions']==r['forbidden']
 labels={};notes={};pending=[]
 for name,reply in q['replies'].items():
  matches=[n for n,x in r['replies'].items() if x==reply]
  values={r['manual_classification'][n] for n in matches}
  if len(values)==1:
   labels[name]=next(iter(values));notes[name]='Inherited exact prompt/gold/reply manual judgment: '+r['note']
  else:labels[name]='pending';notes[name]='Changed or previously inconsistent reply; manual review required';pending.append(name)
 rows.append({'case':case,'input':next(c['input'] for c in f['quality'] if c['id']==case),'gold':q['gold'],'replies':q['replies'],'labels':labels,'notes':notes,'pending':pending})
result={'status':'pending_manual_delta_review','prior_review_sha256':sha(old/'ROOT_QUALITY_REVIEW.json'),'analysis_sha256':sha(D/'analysis.json'),'scope':'High-level reply compatibility only; RAG execution correctness not inferred; baseline is not ground truth','rows':rows}
(D/'QUALITY_REVIEW_PENDING.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'pending_variants':sum(len(r['pending']) for r in rows)},ensure_ascii=False))
