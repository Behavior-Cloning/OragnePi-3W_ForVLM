import sys,json,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;s=json.loads((root/'deploy-spec.json').read_text());base=Path(s['base']);ctx=Path(s['ctx']);tail=Path(s['tail'])
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as src:
  while True:
   b=src.read(4*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
for f in list(base.glob('*.nb'))+[base/'embeddings.f32']:
 q=ctx/f.name
 if not q.exists():q.symlink_to(f)
for f,h in s['expected'].items():assert sha(Path(f))==h,f
out={}
for key,d in [('base',base),('ctx',ctx),('tail',tail)]:
 out[key]={str(f):sha(f) for f in sorted(d.glob('*.nb'))}
 if key!='tail':out[key][str(d/'embeddings.f32')]=sha(d/'embeddings.f32')
r=subprocess.run([str(root/'test-greedy-arm')],capture_output=True,text=True,timeout=60);out['greedy_selftest']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};assert r.returncode==0,out['greedy_selftest']
(root/'verified-graphs.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':'verified','graph_counts':{k:len(out[k]) for k in ['base','ctx','tail']},'selftest':out['greedy_selftest']}))
