from pathlib import Path
import json,hashlib,tarfile,numpy as np
P=Path(__file__).resolve().parent;run='prefixsafe-i16-r2';D=P/'results'/run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((P/'controller.json').read_text());assert s['status']=='complete' and s['returncode']==0
assert json.loads((D/'meta.json').read_text())['status']=='complete'
a=P/(run+'.tar.gz');assert sha(a)==s['archive_sha256']
with tarfile.open(a) as tar:
 for m in tar.getmembers():
  if m.isfile():assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(D/m.name.removeprefix('./'))
for n,h in s['source_sha256'].items():assert sha(P/('runtime/'+n if n=='resident-prefixsafe' else n))==h,n
cfg=json.loads((D/'config.json').read_text());ev=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];variants=[x['name'] for x in cfg['variants']]
closed=[x for x in ev if x['event']=='closed'];assert len(closed)==3 and all(x['metrics']['driver_bytes']==0 for x in closed)
for v,expected in zip(variants,[0,74,444]):
 c=next(x for x in closed if x['variant']==v);assert c['original_prefix_calls']==expected
 log=(D/v/'process.log').read_text();assert log.count('PREFIX_ORIGINAL block=1 ')==expected
 assert log.count('PREFIX_SUFFIX_GROUP blocks=10')==expected
rows=[x for x in ev if x['event']=='result'];prefix=[x for x in ev if x['event']=='prefix_check'];assert len(rows)==216 and len(prefix)==6
for v in variants:
 z=[x for x in rows if x['variant']==v];assert len(z)==72
 assert [sum(x['kind']==k for x in z) for k in ['speed','natural','fixed']]==[16,48,8]
 for e in z:
  m=e['metrics'];assert m['reused_tokens']==e['static_tokens'] and m['new_tokens']==e['dynamic_body_tokens']+8
  assert m['prompt_tokens']==e['static_tokens']+e['dynamic_body_tokens']+8
  assert m['diagnostic']==(e['kind']=='fixed') and m['fixed_feedback']==(e['kind']=='fixed')
  assert e['prepare']['reused_tokens']==0
  if e['kind'] in ['speed','fixed']:assert m['generated_tokens']==(1 if e['kind']=='speed' else 4)
logits={}
for e in [x for x in rows if x['kind']=='fixed']+prefix:
 f=D/e['logits_file'];assert sha(f)==e['logits_sha256'];x=np.fromfile(f,'<f4');assert x.size==e['metrics']['generated_tokens']*151674 and np.isfinite(x).all()
 key=('prefix',e['static_tokens']) if e['event']=='prefix_check' else ('fixed',e['case']);logits[(e['variant'],key)]=x.reshape(-1,151674)
diag=[]
for (v,key),x in logits.items():
 if v==variants[0]:continue
 base=logits[(variants[0],key)];dx=x.astype('float64')-base
 diag.append({'variant':v,'case':key,'byte_equal':bool(np.array_equal(x.view(np.uint32),base.view(np.uint32))),'rel_l2':float(np.linalg.norm(dx)/max(np.linalg.norm(base.astype('float64')),1e-30)),'max_abs':float(abs(dx).max()),'argmax_matches':int((x.argmax(1)==base.argmax(1)).sum()),'steps':len(x)})
base_natural={x['case']:x for x in rows if x['variant']==variants[0] and x['kind']=='natural'};quality={}
for v in variants[1:]:
 z=[x for x in rows if x['variant']==v and x['kind']=='natural'];changed=[]
 for x in z:
  b=base_natural[x['case']]
  if x['metrics']['tokens']!=b['metrics']['tokens']:changed.append({'case':x['case'],'input':x['input'],'baseline':b['reply'],'candidate':x['reply'],'baseline_tokens':b['metrics']['tokens'],'candidate_tokens':x['metrics']['tokens']})
 quality[v]={'cases':len(z),'token_exact':len(z)-len(changed),'changed':changed,'semantic_qualification':'manual_review_pending; parity alone is not task correctness'}
speed=[]
for static in [64,128]:
 for dynamic in [64,128,192,256]:
  for v in variants:
   z=[x['metrics'] for x in rows if x['kind']=='speed' and x['variant']==v and x['static_tokens']==static and x['dynamic_body_tokens']==dynamic];assert len(z)==2
   speed.append({'variant':v,'system_body_tokens':static-5,'system_template_tokens':5,'static_tokens':static,'dynamic_body_tokens':dynamic,'dynamic_template_tokens':8,'new_tokens':dynamic+8,'reused_tokens':static,'output_tokens':1,'repeats':2,'ttft_ms':float(np.median([x['ttft_seconds'] for x in z])*1000),'prefill_ms':float(np.median([x['prefill_seconds']+x['scalar_prompt_seconds'] for x in z])*1000),'under100ms_count':sum(x['ttft_seconds']<=.1 for x in z),'raw_ttft_ms':[x['ttft_seconds']*1000 for x in z]})
resources=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()]
health=[json.loads((D/'pre-health.json').read_text())]+resources+[json.loads((D/'post-health.json').read_text())]
def fields(x,key):return {k:int(v.split()[0]) for k,v in (line.split(':',1) for line in x[key].splitlines())}
def vm(x):return {k:int(v) for k,v in (line.split() for line in x['vmstat'].splitlines())}
def cpu(x):
 z=[int(v) for v in x['stat'].splitlines()[0].split()[1:9]];return sum(z),z[3]+z[4]
a0,i0=cpu(health[0]);a1,i1=cpu(health[-1])
resource_summary={'samples':len(resources),'min_available_MiB':min(fields(x,'meminfo')['MemAvailable']/1024 for x in health),'swapin_pages':vm(health[-1])['pswpin']-vm(health[0])['pswpin'],'swapout_pages':vm(health[-1])['pswpout']-vm(health[0])['pswpout'],'cpu_whole_machine_percent':100*(1-(i1-i0)/max(a1-a0,1)),'swap_used_start_MiB':(fields(health[0],'meminfo')['SwapTotal']-fields(health[0],'meminfo')['SwapFree'])/1024,'swap_used_end_MiB':(fields(health[-1],'meminfo')['SwapTotal']-fields(health[-1],'meminfo')['SwapFree'])/1024}
answer={'resources':resource_summary,'scope':cfg['scope'],'integrity':{'result_requests':216,'prefix_checks':6,'closed_driver_zero':3},'speed':speed,'quality':quality,'logits':diag,'resource_samples':len(resources),'max_temperature_C':max(v for x in resources for v in x['temperatures_mC'].values())/1000,'goal_achieved':False,'source_sha256':{'analyzer':sha(Path(__file__)),'archive':sha(a)}}
(D/'ANALYSIS.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'integrity':answer['integrity'],'quality':{v:{k:x[k] for k in ['cases','token_exact']} for v,x in quality.items()},'speed':speed,'logits':diag,'max_temperature_C':answer['max_temperature_C']},ensure_ascii=False,indent=2))
