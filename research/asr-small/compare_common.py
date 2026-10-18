import json,statistics
from pathlib import Path
from summarize import norm,dist
root=Path(__file__).resolve().parents[2]/'results/asr-small-20261018'
paths=list(root.glob('standalone-*/*/result.json'));q=[p for p in paths if p.parent.name=='qwen']
if not q:raise SystemExit('Qwen pending')
qr=json.loads(q[-1].read_text())['runs'];ids={r['id'] for r in qr};source=json.loads((root/'corpus-manifest.json').read_text());conditions={r['id']:r['condition'] for r in source['samples']};rows=[]
for p in paths:
 d=json.loads(p.read_text());rs={}
 for r in d['runs']:
  if r['id'] in ids:rs.setdefault(r['id'],r)
 groups={}
 for condition in sorted({conditions[i] for i in ids}):
  selected=[r for i,r in rs.items() if conditions[i]==condition];n=sum(len(norm(r['reference'])) for r in selected);err=sum(dist(norm(r['reference']),norm(r['text'])) for r in selected)
  groups[condition]={'n':len(selected),'chars':n,'errors':err,'cer':err/n if n else None,'p50_seconds':statistics.median(r['seconds'] for r in selected),'outputs':[{'id':r['id'],'reference':r['reference'],'text':r['text'],'seconds':r['seconds']} for r in selected]}
 rows.append({'kind':d['kind'],'groups':groups})
(root/'common-qwen-subset.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(rows,ensure_ascii=False,indent=2))
