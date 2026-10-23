import pathlib,json,re,bisect,subprocess,os
P=pathlib.Path(__file__).parent
rows=[json.loads(l) for l in (P/'pali-text-board-results.jsonl').read_text().splitlines() if l.strip()]
tele=json.loads((P/'pali-text-board-telemetry.json').read_text());samples=tele['samples'];hz=tele['clock_ticks'];process_cpu_valid=tele.get('cpu_stat_parser')=='after-final-rparen-v2'
def number(line,key):
 if line is None:return None
 m=re.search(r'(?:^|\s)'+re.escape(key)+r'=(-?[0-9.]+)',line);return float(m[1]) if m else None
def cpu_at(t):
 if not process_cpu_valid:return None
 if not samples or t<samples[0]['t'] or t>samples[-1]['t']:return None
 i=bisect.bisect_left([r['t'] for r in samples],t)
 if i==0:return samples[0]['cpu_ticks']
 a,b=samples[i-1],samples[i];f=(t-a['t'])/(b['t']-a['t']);return a['cpu_ticks']+f*(b['cpu_ticks']-a['cpu_ticks'])
def system_at(t):
 if not samples or t<samples[0]['t'] or t>samples[-1]['t']:return None
 i=bisect.bisect_left([r['t'] for r in samples],t)
 if i==0:return samples[0].get('system_cpu_ticks')
 a,b=samples[i-1],samples[i]
 if 'system_cpu_ticks' not in a or 'system_cpu_ticks' not in b:return None
 f=(t-a['t'])/(b['t']-a['t']);return [x+f*(y-x) for x,y in zip(a['system_cpu_ticks'],b['system_cpu_ticks'])]
def system_interval(begin,end):
 a,b=system_at(begin),system_at(end)
 if a is None or b is None:return dict(system_busy_percent=None,system_iowait_percent=None)
 delta=[y-x for x,y in zip(a,b)];total=sum(delta[:8])
 return dict(system_busy_percent=100*(total-delta[3]-delta[4])/total,system_iowait_percent=100*delta[4]/total)
def resources(sub):
 if not sub:return dict(samples=0)
 return dict(samples=len(sub),rss_max_kb=max(x['rss_kb'] for x in sub),pss_max_kb=max(x['pss_kb'] for x in sub),available_min_kb=min(x['available_kb'] for x in sub),temperature_max=max((v for x in sub for v in x['temps'].values()),default=None),swap_max_kb=max(x['swap_kb'] for x in sub))
def cpu_interval(begin,end):
 a,b=cpu_at(begin),cpu_at(end)
 return None if a is None or b is None else (b-a)/hz/(end-begin)
def cpus(spec):
 out=set()
 for part in spec.split(','):
  if '-' in part:a,b=map(int,part.split('-'));out.update(range(a,b+1))
  elif part:out.add(int(part))
 return out
out=[]
for r in rows:
 if r['event']!='measure':continue
 lines=[x['line'] for x in r['markers']];one=lambda name:next((x for x in lines if x.startswith(name)),None)
 begin=number(one('OPI_REQUEST_BEGIN'),'t');end=number(one('OPI_REQUEST_END'),'t');imstart=begin;imend=begin;first=number(one('OPI_FIRST_TOKEN'),'t')
 toks=[dict(id=int(number(l,'id')),t=number(l,'t')) for l in lines if l.startswith('OPI_TOKEN')];content=[x for x in toks if x['id']!=1];sub=[s for s in samples if begin<=s['t']<=end];cpu=cpu_interval(begin,end)
 has_decode=len(content)>=2
 out.append(dict(case=r['case'],resident_hot=r['resident_hot'],prompt_tokens=int(number(one('OPI_REQUEST_END'),'prompt_tokens')),image_s=imend-imstart,e2e_ttft_s=None if first is None else first-begin,after_vision_ttft_s=None if first is None else first-imend,total_s=end-begin,output_ids=[x['id'] for x in toks],content_tokens=len(content),generated_tokens_including_eos=len(toks),post_first_decode_steps_including_eos=len(toks)-1 if has_decode else None,post_first_decode_steps_excluding_eos=len(content)-1 if has_decode else None,post_first_tokens_s_including_eos=(len(toks)-1)/(toks[-1]['t']-toks[0]['t']) if has_decode else None,post_first_tokens_s_excluding_eos=(len(content)-1)/(content[-1]['t']-content[0]['t']) if has_decode else None,decode_note='single-case post-first average, not distribution' if has_decode else 'one content token plus EOS (or empty output): insufficient content decode intervals',process_cpu_cores=cpu,process_cpu_percent_of_8=None if cpu is None else cpu/8*100,cpu_note=None if cpu is not None else ('invalid original stat parser: gemma comm has spaces, fault fields were sampled; unrecoverable' if not process_cpu_valid else 'continuous samples do not bracket full request; not extrapolated'),**resources(sub),**system_interval(begin,end)))
# Decode from completed token events, not asynchronously captured stdout.
code="import json,sys,sentencepiece as sp; p=sp.SentencePieceProcessor(model_file=sys.argv[1]);print(json.dumps([p.decode([x for x in ids if x!=1]) for ids in json.load(sys.stdin)],ensure_ascii=False))"
r=subprocess.run(['/opt/opi-lab/export-venv/bin/python','-c',code,str(P/'model/tokenizer.model')],input=json.dumps([x['output_ids'] for x in out]),text=True,capture_output=True,env=dict(os.environ,PYTHONPATH=str(P/'python-deps')))
if r.returncode==0:
 for row,text in zip(out,json.loads(r.stdout)):row['decoded_text']=text
else:
 for row in out:row['decoded_text_error']=r.stderr[-500:]
hot=[x for x in out if x['resident_hot'] and x['process_cpu_cores'] is not None];hot_cpu=sum(x['process_cpu_cores']*x['total_s'] for x in hot)/sum(x['total_s'] for x in hot) if hot else None
ready=next((r for r in rows if r['event']=='ready'),None)
loadline=next((x['line'] for x in ready['markers'] if x['line'].startswith('OPI_MODEL_READY')),None) if ready else None
loadend=number(loadline,'t');loadseconds=number(loadline,'load_s');loadcpu=cpu_interval(loadend-loadseconds,loadend) if loadline else None
aff=[dict(pid=pid,allowed=allowed,t=x['t']) for x in samples for pid,allowed in x.get('thread_affinities',{}).items() if not cpus(allowed).issubset({6,7})]
summary=dict(process_cpu_valid=process_cpu_valid,complete=any(r['event']=='complete' and r['exit']==0 for r in rows),errors=[r for r in rows if r['event']=='error'],ready=ready,cases=out,load_cpu_cores=loadcpu,load_cpu_note=None if loadcpu is not None else ('invalid original proc stat parser; CPU unmeasured' if not process_cpu_valid else 'samples did not bracket entire fast mmap model-load; unmeasured'),hot_process_cpu_cores=hot_cpu,hot_process_cpu_percent_of_8=None if hot_cpu is None else hot_cpu/8*100,hot_cpu_cases=[x['case'] for x in hot],telemetry=dict(**resources(samples),guard=tele['guard'],thread_affinity_violations=aff,thread_affinity_sampled=any(x.get('thread_affinities') for x in samples)),system_cpu_definition='Linux /proc/stat first eight fields; busy excludes idle and iowait; guest fields excluded to avoid double count',note='Per-case samples only; not P50/P95. CPU text-only; no vision executed; fixed system KV not implemented.')
(P/'pali-text-board-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
