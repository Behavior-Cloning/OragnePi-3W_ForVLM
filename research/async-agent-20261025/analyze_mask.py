import json,math,statistics,hashlib
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent;r=p/'results/ttft-mask-matrix'
rows=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()];res=[json.loads(x) for x in (r/'resources.jsonl').read_text().splitlines()]
def st(xs):
 xs=sorted(xs);return {'n':len(xs),'p50':statistics.median(xs),'p95':xs[math.ceil(.95*len(xs))-1],'min':min(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {}
variants=['group-base','mask-cache','group-repeat'];base={x['case']:x for x in rows if x['event']=='quality' and x['variant']=='group-base'};diag={x['case']:x for x in rows if x['event']=='diagnostic' and x['variant']=='group-base'}
perf={(x['case'],x['repeat']):x for x in rows if x['event']=='measure' and x['variant']=='group-base'}
for x in rows:
 if x['event']=='diagnostic':
  raw=(r/x['logits_file']).read_bytes()
  assert len(raw)==x['bytes']==4*151674*4
  assert hashlib.sha256(raw).hexdigest()==x['sha256']
  assert x['metrics']['generated_tokens']==4 and x['metrics']['diagnostic'] and x['metrics']['fixed_feedback']
out={'system_text_tokens':81,'system_template_tokens':5,'fixed_prefix_tokens':86,'dynamic_template_tokens':8,'variants':{},'counts':dict(Counter(x['event'] for x in rows)),'complete':[x for x in rows if x['event']=='complete']}
for v in variants:
 q=[x for x in rows if x['event']=='quality' and x['variant']==v];me=[x for x in rows if x['event']=='measure' and x['variant']==v];dd=[x for x in rows if x['event']=='diagnostic' and x['variant']==v]
 z={'performance':{},'quality_token_parity':sum(x['metrics']['tokens']==base[x['case']]['metrics']['tokens'] for x in q),'quality_n':len(q),'performance_token_parity':sum(x['metrics']['tokens']==perf[(x['case'],x['repeat'])]['metrics']['tokens'] for x in me),'diagnostic':[{'case':x['case'],'same_logit_sha':x['sha256']==diag[x['case']]['sha256'],'bytes':x['bytes'],'sha256':x['sha256']} for x in dd],'quality':[{'case':x['case'],'reply':x['reply'],'generated_tokens':x['metrics']['generated_tokens'],'top3':[a['name'] for a in x['top3']]} for x in q],'closed':[x for x in rows if x['event']=='closed' and x['variant']==v]}
 for n in [32,48,64]:
  mm=[x['metrics'] for x in me if x['body_tokens']==n]
  z['performance'][str(n)]={'ttft_s':st([x['ttft_seconds'] for x in mm]),'under_1s':sum(x['ttft_seconds']<1 for x in mm),'new_tokens':sorted({x['new_tokens'] for x in mm}),'reused_tokens':sorted({x['reused_tokens'] for x in mm}),'prompt_tokens':sorted({x['prompt_tokens'] for x in mm}),'output_tokens':sorted({x['generated_tokens'] for x in mm}),'stages':{k:st([x[k] for x in mm]) for k in ['setup_seconds','prefill_seconds','first_head_seconds','scalar_prompt_seconds','batch_calls']}}
 out['variants'][v]=z
cpu=[]
for a,b in zip(res,res[1:]):
 d=[int(y)-int(x) for x,y in zip(a['cpu'][0].split()[1:9],b['cpu'][0].split()[1:9])];t=sum(d)
 if t>0:cpu.append(100*(t-d[3]-d[4])/t)
out['mask_validation']=[x for x in rows if x.get('variant')=='mask-validate']
out['resources']={'system_cpu_percent':st(cpu),'temp_C':st([x['temp_mC']/1000 for x in res]),'host_pss_MiB':st([sum(y.get('Pss',0) for y in x['processes'].values())/1024 for x in res]),'available_MiB':st([x['meminfo']['MemAvailable']/1024 for x in res]),'swap_in_pages':res[-1]['vmstat']['pswpin']-res[0]['vmstat']['pswpin'],'swap_out_pages':res[-1]['vmstat']['pswpout']-res[0]['vmstat']['pswpout']}
out['input_hashes']={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['events.jsonl','resources.jsonl','fixtures.json']}
(r/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for v,z in out['variants'].items():
 print(v,[(n,round(d['ttft_s']['p50'],6),round(d['ttft_s']['p95'],6),d['under_1s']) for n,d in z['performance'].items()], 'quality',z['quality_token_parity'],'diagnostic',z['diagnostic'])
