import json,math,statistics,hashlib,sys
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent;r=p/'results'/(sys.argv[1] if len(sys.argv)>1 else 'ttft-aligned-matrix-v2')
rows=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()];res=[json.loads(x) for x in (r/'resources.jsonl').read_text().splitlines()];fixtures=json.loads((r/'fixtures.json').read_text())
def st(xs):
 xs=sorted(xs);return {'n':len(xs),'p50':statistics.median(xs),'p95':xs[math.ceil(.95*len(xs))-1],'min':min(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {}
def resource(xs):
 cpu=[]
 for a,b in zip(xs,xs[1:]):
  d=[int(y)-int(x) for x,y in zip(a['cpu'][0].split()[1:9],b['cpu'][0].split()[1:9])];t=sum(d)
  if t>0:cpu.append(100*(t-d[3]-d[4])/t)
 return {'system_cpu_percent':st(cpu),'temp_C':st([x['temp_mC']/1000 for x in xs]),'host_pss_MiB':st([sum(y.get('Pss',0) for y in x['processes'].values())/1024 for x in xs]),'available_MiB':st([x['meminfo']['MemAvailable']/1024 for x in xs]),'swap_in_pages':xs[-1]['vmstat']['pswpin']-xs[0]['vmstat']['pswpin'],'swap_out_pages':xs[-1]['vmstat']['pswpout']-xs[0]['vmstat']['pswpout']}
for x in rows:
 if x['event']=='diagnostic':
  raw=(r/x['logits_file']).read_bytes();assert len(raw)==x['bytes']==4*151674*4
  assert hashlib.sha256(raw).hexdigest()==x['sha256']
  assert x['metrics']['generated_tokens']==4 and x['metrics']['diagnostic'] and x['metrics']['fixed_feedback']
variants=['plain','tail8','group','mask','group-repeat'];out={'counts':dict(Counter(x['event'] for x in rows)),'complete':[x for x in rows if x['event']=='complete'],'variants':{},'resources':resource(res),'source_hashes':fixtures['board_source_sha256'],'token_accounting':[{k:s[k] for k in ['static_tokens','system_body_tokens','system_template_tokens']} for s in fixtures['systems']],'speed_output_tokens':1,'dynamic_template_tokens':8}
refq={v:{x['case']:x for x in rows if x['event']=='quality' and x['variant']==v} for v in variants};refd={v:{x['case']:x for x in rows if x['event']=='diagnostic' and x['variant']==v} for v in variants}
for v in variants:
 me=[x for x in rows if x['event']=='measure' and x['variant']==v];qq=list(refq[v].values());dd=list(refd[v].values());ld=next(x for x in rows if x['event']=='loaded' and x['variant']==v);cl=next(x for x in rows if x['event']=='closed' and x['variant']==v)
 z={'performance':{},'quality':[{k:x[k] for k in ['case','static_tokens','body_tokens','input','reply','metrics','top3']} for x in qq],'parity':{ref:{'quality_equal':sum(x['metrics']['tokens']==refq[ref][x['case']]['metrics']['tokens'] for x in qq),'quality_n':len(qq),'diagnostic_equal':sum(x['sha256']==refd[ref][x['case']]['sha256'] for x in dd),'diagnostic_n':len(dd)} for ref in ['plain','group']},'loaded':ld,'closed':cl,'resources':resource([x for x in res if ld['at']<=x['at']<=cl['at']])}
 for static in [64,128]:
  for dynamic in [32,64]:
   xx=[x for x in me if x['static_tokens']==static and x['body_tokens']==dynamic];mm=[x['metrics'] for x in xx]
   assert len(mm)==9 and all(x['reused_tokens']==static and x['new_tokens']==dynamic+8 and x['generated_tokens']==1 for x in mm)
   z['performance'][f's{static}-d{dynamic}']={'system_body_tokens':static-5,'system_template_tokens':5,'static_tokens':static,'dynamic_body_tokens':dynamic,'dynamic_template_tokens':8,'total_prompt_tokens':static+dynamic+8,'reused_tokens':static,'new_tokens':dynamic+8,'generated_tokens':1,'ttft_s':st([x['ttft_seconds'] for x in mm]),'wall_s':st([x['wall_s'] for x in xx]),'under_1s':sum(x['ttft_seconds']<1 for x in mm),'prefix_prepare_s':st([x['prefix_prepare_metrics']['ttft_seconds'] for x in xx]),'stages':{k:st([x[k] for x in mm]) for k in ['setup_seconds','prefill_seconds','first_head_seconds','scalar_prompt_seconds','batch_calls']}}
 out['variants'][v]=z
out['validation_phase']=[x for x in rows if x.get('variant')=='mask-validate'];out['input_hashes']={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['events.jsonl','resources.jsonl','fixtures.json']}
(r/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for v,z in out['variants'].items():
 print(v,[(n,round(d['ttft_s']['p50'],6),round(d['ttft_s']['p95'],6),d['under_1s']) for n,d in z['performance'].items()],z['parity'])
