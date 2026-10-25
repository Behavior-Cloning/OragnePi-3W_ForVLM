#!/usr/bin/env python3
"""Offline analysis of a completed, recovered C1280 original/Conv/original run.
Never connects to the board. Diagnostics are excluded from every speed table.
Semantic judgments require exact previous reviewed text or a hash-bound review.
"""
from pathlib import Path
import argparse, collections, copy, hashlib, importlib.util, json, math, statistics
import numpy as np

P=Path(__file__).resolve().parent
RESEARCH=P.parents[1]
NAMES=['c1280-original','c1280-conv-l2','c1280-repeat']
J=lambda p:json.loads(Path(p).read_text())

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def objhash(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def stats(values):
 xs=sorted(float(x) for x in values)
 if not xs:return {'n':0}
 assert all(math.isfinite(x) for x in xs)
 return {'n':len(xs),'min':xs[0],'p50':statistics.median(xs),
         'p95':xs[math.ceil(.95*len(xs))-1],
         'p95_nearest_rank':xs[math.ceil(.95*len(xs))-1],
         'p95_linear':float(np.quantile(xs,.95,method='linear')),
         'max':xs[-1],'mean':statistics.mean(xs)}
def near(a,b,tol=3e-6):assert abs(a-b)<=tol,(a,b)
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dynamic(m):return m['prefill_seconds']+m['scalar_prompt_seconds']
def differences(a,b):
 same=[i for i,(x,y) in enumerate(zip(a,b)) if x==y]
 return {'exact':a==b,'a_count':len(a),'b_count':len(b),'same_position_count':len(same),
         'first_difference':next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)}
def pair_metrics(a,b):
 am,bm=a['metrics'],b['metrics']
 out={}
 for key,fun in [('ttft_seconds',lambda m:m['ttft_seconds']),('dynamic_prefill_seconds',dynamic),
                 ('decode_tokens_s',lambda m:m['decode_tokens_per_second']),('setup_seconds',lambda m:m['setup_seconds']),
                 ('first_head_seconds',lambda m:m['first_head_seconds'])]:
  av,bv=fun(am),fun(bm)
  out[key]={'a':av,'b':bv,'b_minus_a':bv-av,'b_over_a':bv/av if av else None,
            'speedup_a_over_b':av/bv if bv else None}
 out['prediction_tokens']=differences(am['tokens'],bm['tokens'])
 out['system_prepare_seconds']={'a':a['prefix_prepare_metrics']['total_seconds'],'b':b['prefix_prepare_metrics']['total_seconds']}
 return out

def logit_compare(a,b):
 assert a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all()
 a=a.astype(np.float64);b=b.astype(np.float64);delta=b-a
 steps=[]
 for i,(x,y) in enumerate(zip(a,b)):
  order=np.argsort(-x,kind='stable')[:5];alt=np.argsort(-y,kind='stable')[:5]
  xa=int(order[0]);ya=int(alt[0]);dx=y-x
  # Margins are signed at the baseline winning token as well as native top1-top2.
  yy=y.copy();yy[xa]=-np.inf
  lx=x-(float(np.max(x))+math.log(float(np.exp(x-np.max(x)).sum())))
  ly=y-(float(np.max(y))+math.log(float(np.exp(y-np.max(y)).sum())))
  steps.append({'step':i+1,'argmax_a':xa,'argmax_b':ya,'argmax_equal':xa==ya,
      'top1_margin_a':float(x[order[0]]-x[order[1]]),'top1_margin_b':float(y[alt[0]]-y[alt[1]]),
      'baseline_winner_margin_in_b':float(y[xa]-yy.max()),'top5_a':order.tolist(),'top5_b':alt.tolist(),
      'relative_l2':float(np.linalg.norm(dx)/max(np.linalg.norm(x),1e-30)),
      'max_abs':float(np.abs(dx).max()),'mean_abs':float(np.abs(dx).mean()),
      'KL_a_to_b':float(np.sum(np.exp(lx)*(lx-ly)))})
 return {'float_values_equal':bool(np.array_equal(a,b)),
         'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30)),
         'max_abs':float(np.abs(delta).max()),'mean_abs':float(np.abs(delta).mean()),
         'argmax_equal_count':sum(x['argmax_equal'] for x in steps),'steps':steps}

def proc_ticks(line):
 # Field2 is parenthesized and may itself contain spaces or ')' characters.
 tail=line[line.rfind(')')+2:].split()
 return int(tail[11])+int(tail[12])  # fields14+15; not converted without recorded USER_HZ.
def cpu_fields(row):return [int(v) for v in row['cpu'][0].split()[1:9]]
def intervals(samples):
 out=[]
 for a,b in zip(samples,samples[1:]):
  dt=b['at']-a['at'];d=[v-u for u,v in zip(cpu_fields(a),cpu_fields(b))];total=sum(d)
  assert dt>0 and total>0 and all(v>=0 for v in d)
  out.append({'start':a['at'],'end':b['at'],'duration_s':dt,'same_bucket':(a['variant'],a['phase'])==(b['variant'],b['phase']),
    'variant':b['variant'],'phase':b['phase'],'ticks_total':total,'ticks_busy':total-d[3]-d[4],
    'cpu_percent':100*(total-d[3]-d[4])/total,
    'swapin_pages':b['vmstat']['pswpin']-a['vmstat']['pswpin'],
    'swapout_pages':b['vmstat']['pswpout']-a['vmstat']['pswpout']})
 return out

def resource_bucket(samples,ivs):
 if not samples:return {'samples':0}
 assert all(r['frontend']['error'] is None for r in samples)
 pss=lambda r:sum(p['Pss'] for p in r['processes'].values())/1024
 rss=lambda r:sum(p['VmRSS'] for p in r['processes'].values())/1024
 first,last=samples[0],samples[-1];total=sum(x['ticks_total'] for x in ivs);busy=sum(x['ticks_busy'] for x in ivs)
 rf=first['frontend'];rl=last['frontend'];fdelta=rl['frames']-rf['frames'];odelta=rl['overruns']-rf['overruns']
 return {'samples':len(samples),'observed_span_s':last['at']-first['at'],'cpu_interval_samples':len(ivs),
   'cpu_percent_whole_8cores':stats(x['cpu_percent'] for x in ivs),
   'cpu_weighted_mean_percent_whole_8cores':100*busy/total if total else None,
   'cpu_weighted_equivalent_cores':8*busy/total if total else None,
   'temperature_C':stats(r['temp_mC']/1000 for r in samples),
   'load1':stats(float(r['loadavg'].split()[0]) for r in samples),
   'process_RSS_sum_MiB':stats(rss(r) for r in samples),'process_PSS_sum_MiB':stats(pss(r) for r in samples),
   'process_count_observed':dict(collections.Counter(len(r['processes']) for r in samples)),
   'observed_process_ids':sorted({pid for r in samples for pid in r['processes']}),
   'process_scope':'Explicit Python frontend/RAG/controller PID plus resident Popen PID while worker active; not all system processes or a separate driver-memory sum.',
   'MemAvailable_MiB':stats(r['meminfo']['MemAvailable']/1024 for r in samples),
   'swap_used_MiB':stats((r['meminfo']['SwapTotal']-r['meminfo']['SwapFree'])/1024 for r in samples),
   'swapin_pages_delta_included_intervals':sum(x['swapin_pages'] for x in ivs),
   'swapout_pages_delta_included_intervals':sum(x['swapout_pages'] for x in ivs),
   'frequency_kHz':{c:stats(r['freq_khz'][c] for r in samples if c in r['freq_khz']) for c in sorted({c for r in samples for c in r['freq_khz']})},
   'audio_between_first_last_samples':{'frames':fdelta,'overruns':odelta,'overrun_fraction':odelta/fdelta if fdelta else None,
       'first_max_lag_s':rf['max_lag_s'],'last_max_lag_s':rl['max_lag_s'],'cumulative_lag_new_max_s':rl['max_lag_s']-rf['max_lag_s'],
       'first_max_compute_s':rf['max_compute_s'],'last_max_compute_s':rl['max_compute_s'],
       'stage_wall_seconds_delta':{k:rl['stage_seconds'].get(k,0)-rf['stage_seconds'].get(k,0) for k in rl['stage_seconds']},
       'queue_observed':dict(collections.Counter(r['frontend']['queue'] for r in samples)),
       'caveat':'Lag and max_compute are cumulative maxima; unchanged maximum does not establish zero phase lag. Frames/overruns deltas omit boundary gaps. Queue unit is whole WAV.'}}

def cell_stats(es):
 ms=[e['metrics'] for e in es]
 return {'n':len(es),'cases':[e['case'] for e in es],
    'dynamic_prefill_seconds':stats(dynamic(m) for m in ms),'ttft_seconds':stats(m['ttft_seconds'] for m in ms),
    'caller_observed_first_token_s':stats(e['token_events'][0]['observed_s'] for e in es),
    'decode_tokens_s':stats(31/(m['total_seconds']-m['ttft_seconds']) for m in ms),
    'setup_seconds':stats(m['setup_seconds'] for m in ms),'first_head_seconds':stats(m['first_head_seconds'] for m in ms),
    'residual_TTFT_seconds':stats(m['ttft_seconds']-m['setup_seconds']-dynamic(m)-m['first_head_seconds'] for m in ms),
    'dynamic_prefill_fraction_TTFT':stats(dynamic(m)/m['ttft_seconds'] for m in ms),
    'prefix_prepare':{'total_seconds':stats(e['prefix_prepare_metrics']['total_seconds'] for e in es),
        'prefill_seconds':stats(dynamic(e['prefix_prepare_metrics']) for e in es),'setup_seconds':stats(e['prefix_prepare_metrics']['setup_seconds'] for e in es),
        'excluded_from_hot_TTFT':True,'cache_policy':'reset, prepare only same system text; candidate KV may differ numerically'},
    'api_seconds_whole_request':stats(m['api_seconds'] for m in ms),'CPU_seconds_whole_request':stats(m['cpu_seconds'] for m in ms),
    'C_RSS_request_boundary_MiB':stats(m['rss_kib']/1024 for m in ms),'driver_bytes_request_boundary':stats(m['driver_bytes'] for m in ms),
    'decode_segments':[{'positions':es[0]['decode_segments'][i]['output_positions'],'tokens_s':stats(e['decode_segments'][i]['tokens_s'] for e in es)} for i in range(4)],
    'prefill_le_100ms_count':sum(dynamic(m)<=.1 for m in ms),'ttft_le_100ms_count':sum(m['ttft_seconds']<=.1 for m in ms)}


def main(args):
 from tokenizers import Tokenizer
 from contract import validate_aba, BINARY_SHA
 run=args.run;R=P/'results'/run
 assert R.is_dir(),'No recovered results; this script never queries the board.'
 CTL=J(P/(run+'-controller.json'));C=J(R/'config.json');M=J(R/'meta.json')
 assert CTL['status']=='complete' and CTL['returncode']==0 and M['status']=='complete' and not M['background_errors']
 validate_aba(C)
 E=rows(R/'events.jsonl');RS=rows(R/'resources.jsonl');F=J(R/'fixtures.json');PV=J(R/'provenance.json')
 assert len(RS)>1
 count=collections.Counter(e['event'] for e in E)
 assert {k:count[k] for k in ['measure','quality','diagnostic','loaded','closed','complete']}==dict(measure=72,quality=36,diagnostic=24,loaded=3,closed=3,complete=1),count
 assert not any(count[k] for k in ['error','release_error','audio_error','monitor_error','closed_on_failure','failed'])
 assert [e['variant'] for e in E if e['event']=='loaded']==NAMES
 assert [e['variant'] for e in E if e['event']=='closed']==NAMES
 for n in ['board_fullconv.py','resident_worker_v1.py','context_rag.py','contract.py']:
  assert sha(P/n)==PV[n]==CTL['source_sha256'][n],n
 assert sha(P/'controller_fullconv.py')==C['controller_source_sha256']
 assert sha(P/'inputs.json')==PV['fixture_sha256']==C['expected_fixture_sha256']==CTL['source_sha256']['inputs.json']
 assert sha(P/(run+'-config.json'))==CTL['source_sha256']['config.json']
 assert J(P/(run+'-config.json'))==C
 assert sha(RESEARCH/'resident-agent/agent_replay.py')==PV['frontend_sha256']
 assert sha(RESEARCH/'home-companion-atoms128-v2-20261022/atoms.json')==PV['rag_catalog_sha256']==CTL['source_sha256']['atoms.json']
 assert sha(RESEARCH/'intern-nextopt-20261025/runtime-combined-v2/resident-combined-v2')==BINARY_SHA==CTL['source_sha256']['resident-combined-v2']
 assert sha(C['candidate']['candidate_nb'])==C['candidate']['candidate_sha256']==CTL['source_sha256']['prefill-layer2-conv.nb']
 T=Path(F['tokenizer_path']);assert sha(T)==PV['tokenizer_sha256']==F['tokenizer_sha256']
 tok=Tokenizer.from_file(str(T))
 spec=importlib.util.spec_from_file_location('fullconv_board_readonly',P/'board_fullconv.py')
 bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
 localF=J(P/'inputs.json');bench.attach_system_metadata(localF);assert localF==F
 diag_cases=bench.validate_fixture(copy.deepcopy(F),lambda s:tok.encode(s,add_special_tokens=False).ids)
 CASE={c['id']:c for c in F['performance']+F['quality']}
 perf=[e for e in E if e['event']=='measure'];quality=[e for e in E if e['event']=='quality'];diags=[e for e in E if e['event']=='diagnostic']
 loaded={e['variant']:e for e in E if e['event']=='loaded'};closed={e['variant']:e for e in E if e['event']=='closed'}
 for v in NAMES:
  ld=loaded[v];cl=closed[v]
  assert ld['ready']['layers']==24 and ld['ready']['capacity']==1280
  assert not ld['ready']['true_tail8_enabled'] and ld['ready']['single_pass_greedy_enabled']
  assert ld['binary_sha256']==BINARY_SHA and cl['metrics']['driver_bytes']==0
  assert len([e for e in perf if e['variant']==v])==24 and len([e for e in quality if e['variant']==v])==12 and len([e for e in diags if e['variant']==v])==8
 G=J(R/'graph-provenance.json');assert [g['variant'] for g in G]==NAMES
 for g,v in zip(G,C['variants']):
  assert len(g['files'])==29
  assert {r['declared_path']:r['sha256'] for r in g['files']}==v['expected_graph_sha256']
  assert all(r['sha256']==r['expected_sha256'] for r in g['files'])
  for row in g['files']:
   name=Path(row['declared_path']).name
   want=C['overlay_contract']['symlinks'][name] if v['name']==NAMES[1] else row['declared_path']
   assert row['resolved_path']==want,(v['name'],name)
 overlay=J(R/'overlay-provenance.json');assert overlay==CTL['overlay_verified_before_launch']
 assert len(overlay['files'])==29
 for r in overlay['files']:
  assert r['link_target']==r['resolved_path']==C['overlay_contract']['symlinks'][r['name']]
 assert next(r for r in overlay['files'] if r['name']=='prefill-layer2.nb')['bytes']==C['candidate']['candidate_bytes']

 for e in perf+quality+diags:
  c=CASE[e['case']];m=e['metrics'];kind=e['event']
  assert m['prompt_tokens']==len(c['tokens']) and m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8
  assert m['prompt_tokens']==m['reused_tokens']+m['new_tokens'] and m['kv_tokens']==m['prompt_tokens']+m['generated_tokens']-1
  assert not m['true_tail8_calls'] and len(m['tokens'])==m['generated_tokens']
  assert m['diagnostic']==(kind=='diagnostic') and m['fixed_feedback']==(kind!='quality')
  assert all(math.isfinite(m[k]) and m[k]>=0 for k in ['prefill_seconds','scalar_prompt_seconds','ttft_seconds','total_seconds','setup_seconds','first_head_seconds'])
  assert m['batch_calls']*16-max(0,16-m['tail_rows']) * int(m['tail_rows']>0)+m['scalar_prompt_tokens']==m['new_tokens']
  if kind=='measure':
   assert e['repeat']==0 and m['generated_tokens']==32 and e['feedback_tokens']==C['feedback_tokens'] and e['speed_profileflag']=='none'
   assert e['prompt_sha256']==c['tokens_sha256'];near(e['dynamic_prefill_seconds'],dynamic(m))
   assert m['batch_calls']==math.ceil(m['new_tokens']/16) and m['scalar_prompt_tokens']==0 and m['tail_rows']==8
  if kind=='diagnostic':assert m['generated_tokens']==4 and not e['timing_eligible']
  else:
   prep=e['prefix_prepare_metrics'];assert prep['prompt_tokens']==prep['new_tokens']==c['static_tokens'] and prep['reused_tokens']==0 and prep['generated_tokens']==1
   assert not prep['diagnostic'] and not prep['fixed_feedback'] and prep['kv_tokens']==c['static_tokens']
   assert prep['batch_calls']==c['static_tokens']//16 and prep['scalar_prompt_tokens']==0 and prep['tail_rows']==0
   assert e['system_body_tokens']==c['static_tokens']-5 and e['system_template_tokens']==5
   te=e['token_events'];assert len(te)==m['generated_tokens'] and [x['token'] for x in te]==m['tokens'];near(te[0]['elapsed'],m['ttft_seconds'])
   assert all(b['elapsed']>a['elapsed'] for a,b in zip(te,te[1:]))
   if m['generated_tokens']>1:near(m['decode_tokens_per_second'],(m['generated_tokens']-1)/(m['total_seconds']-m['ttft_seconds']),2e-4)
   if kind=='measure':
    assert sum(s['intervals'] for s in e['decode_segments'])==31
    near(sum(s['seconds'] for s in e['decode_segments']),te[-1]['elapsed']-te[0]['elapsed'])
   else:
    assert tok.decode(m['tokens'])==e['reply'] and e['output_tokens_including_eos']==m['generated_tokens']
    assert e['output_tokens_excluding_terminal_eos']==m['generated_tokens']-int(m['tokens'][-1]==C['defaults']['eos'])
    assert len(e['top3'])==3 and all(a['mock_only'] and not a['enabled_on_board'] for a in e['top3'])

 cells=[];perf_index={(e['variant'],e['case']):e for e in perf};assert len(perf_index)==72
 for s in [64,128]:
  for d in [64,128,192,256]:
   cell={'static_tokens':s,'system_body_tokens':s-5,'system_template_tokens':5,'dynamic_body_tokens':d,'dynamic_template_tokens':8,'actual_new_tokens':d+8,'prompt_tokens':s+d+8,'output_steps':32,'variants':{}}
   for v in NAMES:
    es=[e for e in perf if e['variant']==v and e['static_tokens']==s and e['body_tokens']==d]
    assert len(es)==3 and {e['case'] for e in es}=={f's{s}-d{d}-{i}' for i in range(3)}
    cell['variants'][v]=cell_stats(es)
   cells.append(cell)
 pairs=[]
 for c in F['performance']:
  a,b,z=[perf_index[(v,c['id'])] for v in NAMES]
  p={'case':c['id'],'static_tokens':c['static_tokens'],'dynamic_body_tokens':c['body_tokens'],
     'candidate_vs_original':pair_metrics(a,b),'repeat_vs_original':pair_metrics(a,z),'candidate_vs_repeat':pair_metrics(z,b)}
  a0,a1,bv=a['metrics']['ttft_seconds'],z['metrics']['ttft_seconds'],b['metrics']['ttft_seconds']
  p['TTFT_B_less_than_both_A']=bv<min(a0,a1)
  p['TTFT_B_more_than_both_A']=bv>max(a0,a1)
  p['prefill_B_less_than_both_A']=dynamic(b['metrics'])<min(dynamic(a['metrics']),dynamic(z['metrics']))
  p['midpoint_A_reference']={'TTFT_speedup':(a0+a1)/2/bv,
      'prefill_speedup':(dynamic(a['metrics'])+dynamic(z['metrics']))/2/dynamic(b['metrics']),
      'note':'Arithmetic midpoint of sequential A measurements; useful drift bracket, not randomized or time-interpolated causal estimate.'}
  pairs.append(p)

 diagnostic_files=[];arrays={};dindex={}
 for v in NAMES:assert {e['case'] for e in diags if e['variant']==v}=={c['id'] for c in diag_cases}
 for e in diags:
  f=(R/e['logits_file']).resolve();assert R.resolve() in f.parents
  assert f.stat().st_size==e['bytes']==4*151674*4 and sha(f)==e['sha256']
  arr=np.fromfile(f,dtype='<f4').reshape(4,151674);assert np.isfinite(arr).all()
  assert arr.argmax(1).tolist()==e['metrics']['tokens']
  assert e['metrics']['tokens']==perf_index[(e['variant'],e['case'])]['metrics']['tokens'][:4]
  key=(e['variant'],e['case']);assert key not in arrays;arrays[key]=arr;dindex[key]=e
  diagnostic_files.append({'variant':e['variant'],'case':e['case'],'file':e['logits_file'],'bytes':e['bytes'],'sha256':sha(f),'finite':True,'argmax_matches_prediction':True,'matches_timed_first4':True})
 diagnostic_pairs=[]
 for c in diag_cases:
  cid=c['id'];result={'case':cid,'system_tokens':c['static_tokens'],'dynamic_body_tokens':c['body_tokens'],'actual_new_tokens':c['body_tokens']+8}
  for title,va,vb in [('B_vs_A0',NAMES[0],NAMES[1]),('A1_vs_A0',NAMES[0],NAMES[2]),('B_vs_A1',NAMES[2],NAMES[1])]:
   out=logit_compare(arrays[(va,cid)],arrays[(vb,cid)])
   out['bytes_equal']=dindex[(va,cid)]['sha256']==dindex[(vb,cid)]['sha256']
   result[title]=out
  diagnostic_pairs.append(result)

 ivs=intervals(RS);resources=resource_bucket(RS,ivs)
 resources.update(sample_gap_s=stats(b['at']-a['at'] for a,b in zip(RS,RS[1:])),
    swapin_pages_total=RS[-1]['vmstat']['pswpin']-RS[0]['vmstat']['pswpin'],swapout_pages_total=RS[-1]['vmstat']['pswpout']-RS[0]['vmstat']['pswpout'],
    swap_page_size_note='Counts are pages. 4096-byte conversion is reported conditionally, not newly measured by this harness.',
    temperature_scope='thermal_zone0 only',driver_scope='ready/request/normal-close boundaries; no continuous driver sample; do not add driver mapped memory to PSS blindly')
 resources['swapout_MiB_if_page4096']=resources['swapout_pages_total']/256
 resources['variants']={}
 for v in NAMES:
  ss=[r for r in RS if r['variant']==v];vv=[x for x in ivs if x['variant']==v and x['same_bucket']]
  rr=resource_bucket(ss,vv)
  rr['loaded_driver_bytes']=loaded[v]['ready']['driver_bytes'];rr['closed_driver_bytes']=closed[v]['metrics']['driver_bytes']
  rr['load_seconds']=loaded[v]['ready']['load_seconds']
  rr['phases']={ph:resource_bucket([r for r in ss if r['phase']==ph],[x for x in vv if x['phase']==ph]) for ph in sorted({r['phase'] for r in ss})}
  resources['variants'][v]=rr
 resources['boundary_intervals']=[x for x in ivs if not x['same_bucket']]
 resources['interval_accounting']='Global includes all adjacent intervals. Variant/phase buckets include only intervals wholly within same variant and phase; cross-boundary deltas remain listed separately.'
 resources['process_CPU_ticks_observed']={pid:max(proc_ticks(r['processes'][pid]['stat']) for r in RS if pid in r['processes'])-min(proc_ticks(r['processes'][pid]['stat']) for r in RS if pid in r['processes']) for pid in sorted({pid for r in RS for pid in r['processes']})}
 audio=copy.deepcopy(M['audio']);assert audio['error'] is None
 audio.update(overrun_fraction=audio['overruns']/audio['frames'],effective_audio_seconds=audio['frames']*.016,
    endpoint_count=count['audio_endpoint'],first_sample_max_lag_s=RS[0]['frontend']['max_lag_s'],last_sample_max_lag_s=RS[-1]['frontend']['max_lag_s'],
    scope='Prerecorded WAV frontend runs concurrently; requests are independent fixtures, not triggered by speech endpoint. No speech-end-to-token latency in this experiment.',
    overrun_definition='Frame compute >16ms; not proof of lost frames.',queue_unit='one whole WAV',stage_seconds_unit='accumulated stage wall time, not CPU seconds or per-utterance latency')


 prior_path=P.parent/'fullbaseline/results/fullbaseline-c1280-r1/analysis.json'
 prior=J(prior_path) if prior_path.exists() else None
 prior_quality={x['case']:x for x in prior['natural_quality']} if prior else {}
 qindex={(e['variant'],e['case']):e for e in quality};assert len(qindex)==36
 supplied=J(args.reviews).get('assessments',[]) if args.reviews else []
 review_index={(x['variant'],x['case']):x for x in supplied};assert len(review_index)==len(supplied)
 review=[]
 for c in F['quality']:
  base=qindex[(NAMES[0],c['id'])]
  for v in NAMES:
   e=qindex[(v,c['id'])];m=e['metrics'];qnames=[a['name'] for a in e['top3']]
   binding={'reply_sha256':objhash(e['reply']),'top3_sha256':objhash([{'id':a['id'],'name':a['name']} for a in e['top3']])}
   r={'variant':v,'case':c['id'],'input':c['input'],'frozen_constraint':c['task_constraint_for_manual_audit'],
      'system_tokens':c['static_tokens'],'system_body_tokens':c['static_tokens']-5,'system_template_tokens':5,
      'dynamic_body_tokens':c['body_tokens'],'dynamic_template_tokens':8,'actual_new_tokens':c['body_tokens']+8,
      'reply':e['reply'],'top3':e['top3'],'finish_reason':m['finish_reason'],
      'output_tokens_including_eos':e['output_tokens_including_eos'],'output_tokens_excluding_eos':e['output_tokens_excluding_terminal_eos'],
      'tokens_vs_A0':differences(base['metrics']['tokens'],m['tokens']),
      'top3_ids_vs_A0_exact':[x['id'] for x in base['top3']]==[x['id'] for x in e['top3']],
      'TTFT_seconds':m['ttft_seconds'],'dynamic_prefill_seconds':dynamic(m),'rag_seconds':e['rag_s'],
      'task_following_pass':None,'task_assessment':'pending semantic review','rag_contains_conflicting_action':None,
      'judgment_method':'pending','independent_holdout':c['independent_holdout'],**binding}
   pq=prior_quality.get(c['id'])
   if pq and pq['input']==c['input'] and pq['expected']==c['task_constraint_for_manual_audit']:
    if pq['reply']==e['reply']:
     r.update(task_following_pass=pq['task_following_pass_manual'],task_assessment=pq['assessment'],
              judgment_method='Exact same reply, input and criterion as previously manually reviewed baseline; not a new independent judgment.')
    if pq['top3']==qnames:
     r['rag_contains_conflicting_action']=pq['top3_semantic_conflict']
     r['rag_judgment_method']='Exact same ranked action names under same frozen user constraint as previous manual baseline review.'
   if (v,c['id']) in review_index:
    rev=review_index[(v,c['id'])]
    assert all(rev[k]==binding[k] for k in binding),'Review does not bind to actual text/ranked actions'
    assert isinstance(rev['task_following_pass'],bool) and isinstance(rev['rag_contains_conflicting_action'],bool)
    assert rev['task_assessment'] and rev['rag_assessment']
    r.update({k:rev[k] for k in ['task_following_pass','task_assessment','rag_contains_conflicting_action','rag_assessment']})
    r['judgment_method']='Explicit hash-bound semantic review supplied to analyzer'
   r['pending_semantic_review']=r['task_following_pass'] is None or r['rag_contains_conflicting_action'] is None
   review.append(r)
 quality_summary={v:{'n':12,'task_pass':sum(r['task_following_pass'] is True for r in review if r['variant']==v),
                     'task_fail':sum(r['task_following_pass'] is False for r in review if r['variant']==v),
                     'pending':sum(r['pending_semantic_review'] for r in review if r['variant']==v),
                     'RAG_conflict_lists':sum(r['rag_contains_conflicting_action'] is True for r in review if r['variant']==v),
                     'exact_prediction_sequences_vs_A0':sum(r['tokens_vs_A0']['exact'] for r in review if r['variant']==v),
                     'same_ranked_RAG_ids_vs_A0':sum(r['top3_ids_vs_A0_exact'] for r in review if r['variant']==v),
                     'RAG_seconds':stats(r['rag_seconds'] for r in review if r['variant']==v)} for v in NAMES}
 summary={'status':'completed_experiment_analysis_not_deployment_approval','integrity_pass':True,
    'scope':M['scope'],'counts':dict(count),'variants_in_order':NAMES,
    'source_provenance_verified':True,'actual_fixture_retokenization_verified':True,'graph_records_verified':87,
    'graph_hash_limit':'Recomputed local source/candidate hashes and checked recovered board hash records; analyst never accessed board.',
    'candidate_sha256':C['candidate']['candidate_sha256'],'binary_sha256':BINARY_SHA,
    'quantization_scope':'Only original FP16 down FC implementation becomes FP16 Conv inside fused prefill layers2/3. Same quantizer/weight payload; not an INT4/W8 precision change. Decoder/head/embedding unchanged.',
    'cells':cells,'paired_cases':pairs,'diagnostic_files':diagnostic_files,'diagnostic_comparisons':diagnostic_pairs,
    'diagnostic_scope':'4 full-vocabulary steps x8 fixed long-body inputs x3 variants; same frozen feedback. Compared to production NPU A0/A1, not a full FP32 oracle.',
    'diagnostic_summary':{
      title:{'files':len(diagnostic_pairs),'bitwise_equal_files':sum(x[title]['bytes_equal'] for x in diagnostic_pairs),
             'argmax_equal_steps':sum(x[title]['argmax_equal_count'] for x in diagnostic_pairs),'total_steps':32,
             'relative_l2':stats(x[title]['relative_l2'] for x in diagnostic_pairs),'max_abs':stats(x[title]['max_abs'] for x in diagnostic_pairs)}
      for title in ['B_vs_A0','A1_vs_A0','B_vs_A1']},
    'quality':review,'quality_summary':quality_summary,'quality_manual_baseline_source':str(prior_path) if prior else None,
    'quality_manual_baseline_sha256':sha(prior_path) if prior else None,
    'quality_scope':'6 existing short household instructions x2 systems x3 variants. Not long-body64..256 natural quality. Safe omission fails requested task; token parity is not task success. Mock Top3 does not authorize execution.',
    'resources':resources,'audio':audio,'controller_seconds':CTL['ended']-CTL['started'],
    'analyzer_sha256':sha(Path(__file__)),'quantile_policy':{'primary':'nearest-rank P95; n3 equals observed maximum','secondary':'NumPy linear interpolation P95 separately; never mix with primary'},
    'raw_sha256':{n:sha(R/n) for n in ['events.jsonl','resources.jsonl','fixtures.json','config.json','provenance.json','graph-provenance.json','overlay-provenance.json','meta.json']},
    'limitations':[
       'n3 per variant/cell is a developer sample, not a stable production P95 or 100ms SLO.',
       'A-B-A is sequential, not randomized; later A brackets drift but cannot eliminate confounds.',
       'Fixed32 feedback tests 32 prediction calls, not 32 autoregressively generated answer tokens.',
       'System KV freshly prepared before each request; prep excluded from hot TTFT but contributes test load. Candidate system KV may differ numerically.',
       'TTFT starts after runtime receives/validates request; tokenization, queueing, system prep and speech endpoint excluded.',
       'No NPU profiling in speed samples; full-logit diagnostic readback excluded from speed.',
       'No image encoder, microphone or physical action; prerecorded speech, RAG queries only during natural quality.',
       'Short run cannot establish long-term leak-free stability. Driver ready/request/close only; PSS not all-system memory.',
       'Lag/max_compute counters cumulative; phase attribution limited to sampled changes, not fresh phase maxima.',
       '20 frozen boundary cases not executed in this A-B-A matrix.']}
 summary['paired_overview']={
    'candidate_TTFT_faster_than_both_A_count':sum(p['TTFT_B_less_than_both_A'] for p in pairs),
    'candidate_TTFT_slower_than_both_A_count':sum(p['TTFT_B_more_than_both_A'] for p in pairs),
    'candidate_prefill_faster_than_both_A_count':sum(p['prefill_B_less_than_both_A'] for p in pairs),
    'candidate_fixed32_exact_vs_A0':sum(p['candidate_vs_original']['prediction_tokens']['exact'] for p in pairs),
    'repeat_fixed32_exact_vs_A0':sum(p['repeat_vs_original']['prediction_tokens']['exact'] for p in pairs),
    'repeat_TTFT_ratio':stats(p['repeat_vs_original']['ttft_seconds']['b_over_a'] for p in pairs),
    'candidate_midpoint_TTFT_speedup':stats(p['midpoint_A_reference']['TTFT_speedup'] for p in pairs),
    'candidate_midpoint_prefill_speedup':stats(p['midpoint_A_reference']['prefill_speedup'] for p in pairs)}
 (R/'analysis-fullconv.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 template={'purpose':'Read actual reply and Top3 under frozen input. Safe omission fails task. Hashes prevent stale judgments.',
    'assessments':[{k:r[k] for k in ['variant','case','input','frozen_constraint','reply','top3','reply_sha256','top3_sha256','task_following_pass','task_assessment','rag_contains_conflicting_action']} | {'rag_assessment':r.get('rag_assessment','pending explicit rationale'),'pending':r['pending_semantic_review']} for r in review]}
 (R/'quality-review-template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2)+'\n')
 write_report(R,summary)
 print(json.dumps({'status':summary['status'],'integrity_pass':True,'counts':dict(count),'paired_overview':summary['paired_overview'],
    'diagnostic':summary['diagnostic_summary'],'quality_summary':quality_summary,'closed_driver_bytes':[closed[v]['metrics']['driver_bytes'] for v in NAMES],
    'analysis':str(R/'analysis-fullconv.json')},ensure_ascii=False))


def write_report(R,s):
 lines=['# C1280 单 L2 FP16 Conv：A–B–A 离线结果分析','',
   'A0原版，B仅替换融合层2/3的prefill-layer2，A1再跑原版。相同binary、量化精度、权重、输入、固定反馈和flags。候选system KV也经过新图，不保证跨版本数值相同。','',
   'System总token64/128（正文59/123＋模板5）；动态正文64/128/192/256＋模板8，实际新增72/136/200/264。只复用system，性能输出固定32次预测。72条性能、36条自然回复、24份各4步诊断分开统计。','',
   '主表P95采用nearest-rank，每格3条，等于观察最大值；JSON另列linear P95，不混用。小样本不构成稳定SLO。prefill包含batch＋scalar动态prompt工作；TTFT不含tokenization、排队、system准备或语音判停。','',
   '|System总|动态正文/新增|prefill A0/B/A1 P50秒|TTFT A0/B/A1 P50秒|B TTFT P95秒|B decode P50 token/s|','|---:|---:|---:|---:|---:|---:|']
 for c in s['cells']:
  vv=[c['variants'][v] for v in NAMES];pf='/'.join(f"{x['dynamic_prefill_seconds']['p50']:.4f}" for x in vv);tf='/'.join(f"{x['ttft_seconds']['p50']:.4f}" for x in vv)
  lines.append(f"|{c['static_tokens']}|{c['dynamic_body_tokens']}/{c['actual_new_tokens']}|{pf}|{tf}|{vv[1]['ttft_seconds']['p95']:.4f}|{vv[1]['decode_tokens_s']['p50']:.3f}|")
 o=s['paired_overview']
 lines+=['',f"逐case：B的TTFT比前后A都低{o['candidate_TTFT_faster_than_both_A_count']}/24，prefill比两次A都低{o['candidate_prefill_faster_than_both_A_count']}/24。A1/A0 TTFT比值中位{o['repeat_TTFT_ratio']['p50']:.4f}，范围{o['repeat_TTFT_ratio']['min']:.4f}～{o['repeat_TTFT_ratio']['max']:.4f}。逐条差值、漂移、setup/首head/剩余耗时及decode分段见JSON。",'',
   '## System准备与数值','',
   '每次reset并单独重建system，然后动态请求只处理新增token。准备不计热TTFT但计负载；不同于生产长期复用同一system的总负载。','',
   '|System总|动态正文|system准备 A0/B/A1 P50秒|','|---:|---:|---:|']
 for c in s['cells']:
  vals='/'.join(f"{c['variants'][v]['prefix_prepare']['total_seconds']['p50']:.4f}" for v in NAMES)
  lines.append(f"|{c['static_tokens']}|{c['dynamic_body_tokens']}|{vals}|")
 lines+=['',f"固定反馈32步整条预测：B与A0相同{o['candidate_fixed32_exact_vs_A0']}/24；A1与A0相同{o['repeat_fixed32_exact_vs_A0']}/24。这不是自然生成正确率。",'',
   '|诊断比较|f32逐位相同|argmax相同|全文件相对L2最大|最大绝对误差|','|---|---:|---:|---:|---:|']
 for name,x in s['diagnostic_summary'].items():
  lines.append(f"|{name}|{x['bitwise_equal_files']}/8|{x['argmax_equal_steps']}/32|{x['relative_l2']['max']:.8g}|{x['max_abs']['max']:.8g}|")
 lines+=['','24份实际f32均重算SHA、长度、finite和argmax；每份2,426,784字节。逐步top1 margin、原赢家在候选中的有符号margin、top5、KL和误差见JSON。对照是原NPU实现，不是全模型FP32 oracle。','',
   '## 短自然任务和RAG','',
   '12条是6个已有家庭指令×2 system，不是长正文64～256自然质量。保持静止却遗漏要求的点头/摆臂/弯腰算任务失败。仅输入、约束、回复完全相同时沿用前次逐条语义审核；新内容单独审。','',
   '|阶段|任务通过/失败/待审|自然预测与A0相同|RAG含冲突列表|','|---|---:|---:|---:|']
 for v in NAMES:
  q=s['quality_summary'][v];lines.append(f"|{v}|{q['task_pass']}/{q['task_fail']}/{q['pending']}|{q['exact_prediction_sequences_vs_A0']}/12|{q['RAG_conflict_lists']}/12|")
 lines+=['','检索动作均mock未执行。Top3有方向或禁止约束冲突，不能因相关分高就一起执行；原RAG问题不由算子优化解决。完整输入/回复/Top3和hash在JSON及quality-review-template.json。','',
   '## 负载与语音','',
   '|阶段性能采样|整机CPU均值%|最高温°C|PSS和峰值MiB|最低可用MiB|阶段内交换出页|','|---|---:|---:|---:|---:|---:|']
 for v in NAMES:
  p=s['resources']['variants'][v]['phases']['performance']
  lines.append(f"|{v}|{p['cpu_weighted_mean_percent_whole_8cores']:.2f}|{p['temperature_C']['max']:.3f}|{p['process_PSS_sum_MiB']['max']:.2f}|{p['MemAvailable_MiB']['min']:.2f}|{p['swapout_pages_delta_included_intervals']}|")
 rs=s['resources'];au=s['audio']
 lines+=['',f"全轮{rs['samples']}条，最高{rs['temperature_C']['max']:.3f}°C（thermal_zone0），8核整机CPU加权均值{rs['cpu_weighted_mean_percent_whole_8cores']:.2f}%。交换入/出页增量{rs['swapin_pages_total']}/{rs['swapout_pages_total']}；若4096字节/页，交换出{rs['swapout_MiB_if_page4096']:.2f}MiB。频率、load、PID覆盖、跨阶段区间见JSON。三次关闭driver均0；PSS/driver映射可能重叠，不能相加。",'',
   f"语音{au['frames']}帧、{au['endpoint_count']}个endpoint；{au['overruns']}帧计算>16ms（{100*au['overrun_fraction']:.2f}%），不等于丢帧。累计最大lag {1000*au['max_lag_s']:.2f}ms，最大帧计算{1000*au['max_compute_s']:.2f}ms。阶段仅提供累计计数增量，历史最大lag不是阶段独立测值。",'',
   '语音为预录WAV逐帧流式回放，降噪/VAD/KWS/ASR并发；RAG只在自然阶段查询，其余常驻。LLM fixture请求不由语音endpoint触发，不能据此计算用户说完→首token。','',
   '## 限制','',
   '实验完整不等于允许替换现役图。非逐位等价须结合语义和数值审查；短试验不能证明长期稳定、零泄漏或100ms目标。冻结边界20例未执行。']
 (R/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')


def self_test():
 assert stats([1,2,3])['p95']==3 and stats([1,2,3])['p95_linear']==2.9
 x=np.array([[0,3,3,-2],[0,0,1,-5]],dtype=np.float32)
 a=logit_compare(x,x);assert a['float_values_equal'] and a['argmax_equal_count']==2 and a['relative_l2']==0
 y=x.copy();y[0,2]=3.25;b=logit_compare(x,y)
 assert b['argmax_equal_count']==1 and b['steps'][0]['argmax_a']==1 and b['steps'][0]['argmax_b']==2
 assert b['steps'][0]['top1_margin_a']==0 and b['steps'][0]['baseline_winner_margin_in_b']==-.25
 assert differences([1,2],[1,2])['first_difference'] is None
 assert differences([1],[1,2])['first_difference']==1
 line='77 (model worker ) 0) S '+' '.join(str(i) for i in range(4,53))
 assert proc_ticks(line)==14+15
 assert stats([])=={'n':0}
 # Existing real baseline exercises helper schemas only, never relabeled as Conv.
 old=P.parent/'fullbaseline/results/fullbaseline-c1280-r1';tested=[]
 if old.exists():
  es=rows(old/'events.jsonl');ss=rows(old/'resources.jsonl');rr=resource_bucket(ss,intervals(ss))
  assert rr['samples']==len(ss) and 0<=rr['cpu_weighted_mean_percent_whole_8cores']<=100
  mm=[e for e in es if e['event']=='measure'][:3];assert cell_stats(mm)['n']==3
  dd=next(e for e in es if e['event']=='diagnostic');ar=np.fromfile(old/dd['logits_file'],dtype='<f4').reshape(4,151674)
  assert logit_compare(ar,ar)['argmax_equal_count']==4
  tested=['existing baseline telemetry schema','existing baseline cell schema','full-vocabulary identity comparison']
 print(json.dumps({'status':'helper_selftests_passed_not_a_new_experiment','synthetic_tests':['nearest rank vs linear','earliest tie','changed winner and signed margin','length mismatch','proc stat spaces/parentheses'],'read_only_baseline_schema_tests':tested},ensure_ascii=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run',nargs='?',default='fullconv-c1280-aba-r1');ap.add_argument('--reviews');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.self_test:self_test()
 else:main(args)
