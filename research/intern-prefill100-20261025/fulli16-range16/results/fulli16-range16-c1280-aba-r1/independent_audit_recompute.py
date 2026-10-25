import pathlib,json,hashlib,struct,collections,math,numpy as np
T=pathlib.Path('/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/intern-prefill100-20261025')
P=T/'fulli16-range16'; run='fulli16-range16-c1280-aba-r1'; D=P/'results'/run
J=lambda p:json.loads(p.read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x]
stat=lambda x:{'n':len(x),'min':float(np.min(x)),'p50':float(np.median(x)),'p95_linear':float(np.quantile(x,.95)),'p95_nearest_rank':float(np.sort(x)[math.ceil(.95*len(x))-1]),'max':float(np.max(x))}
ctl=J(P/(run+'-controller.json')); assert ctl['status']=='complete' and ctl['returncode']==0
cfg=J(D/'config.json');meta=J(D/'meta.json');ev=rows(D/'events.jsonl');rs=rows(D/'resources.jsonl');fi=J(D/'fixtures.json');prov=J(D/'provenance.json')
assert meta['status']=='complete' and not meta['background_errors']
counts=collections.Counter(x['event'] for x in ev); names=['c1280-original','c1280-i16-range16-l2','c1280-repeat']
for k,n in {'measure':72,'quality':72,'diagnostic':24,'loaded':3,'closed':3,'complete':1,'graph_verified':3}.items():assert counts[k]==n,(k,counts[k])
assert not any(counts[k] for k in ['error','failed','audio_error','monitor_error','release_error','closed_on_failure'])
assert J(P/(run+'-config.json'))==cfg and sha(P/(run+'-config.json'))==ctl['source_sha256']['config.json']
assert sha(P/'controller_fulli16.py')==cfg['controller_source_sha256']
for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py']:assert sha(P/n)==prov[n]==ctl['source_sha256'][n]
assert sha(P/'inputs.json')==prov['fixture_sha256']==cfg['expected_fixture_sha256']==ctl['source_sha256']['inputs.json']
assert fi['performance']==J(T/'fullbaseline/inputs.json')['performance']
assert fi['quality']==J(T/'fullconv-quality/semantic-inputs.json')['quality']
fp={c['id']:c for c in fi['performance']};fq={c['id']:c for c in fi['quality']};fc={**fp,**fq}
th=lambda ids:hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()
for c in fc.values():
 assert th(c['tokens'])==c['tokens_sha256'] and th(c['prefix_tokens'])==c['prefix_sha256']
 assert c['tokens'][:c['static_tokens']]==c['prefix_tokens'] and len(c['tokens'])==c['static_tokens']+c['body_tokens']+8
 assert c['system_body_tokens']==c['static_tokens']-5
 if 'body_token_ids' in c:assert len(c['body_token_ids'])==c['body_tokens'] and th(c['body_token_ids'])==c['body_tokens_sha256']
source_maps={}
for e in ev:
 if e['event']=='graph_verified':
  assert len(e['files'])==29
  for x in e['files']:assert x['sha256']==x['expected_sha256']
  source_maps[e['variant']]={pathlib.Path(x['declared_path']).name:x['sha256'] for x in e['files']}
assert source_maps[names[0]]==source_maps[names[2]]
changed=[k for k in source_maps[names[0]] if source_maps[names[0]][k]!=source_maps[names[1]][k]]
assert changed==['prefill-layer2.nb'],changed
assert source_maps[names[1]]['prefill-layer2.nb']==cfg['candidate']['candidate_sha256']==sha(pathlib.Path(cfg['candidate']['candidate_nb']))
loads=[e for e in ev if e['event']=='loaded'];closes=[e for e in ev if e['event']=='closed'];assert [e['variant'] for e in loads]==names==[e['variant'] for e in closes]
for e in loads:
 assert e['binary_sha256']=='ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d'
 assert e['ready']['layers']==24 and e['ready']['capacity']==1280 and not e['ready']['true_tail8_enabled'] and e['ready']['single_pass_greedy_enabled']
for e in closes:assert e['metrics']['driver_bytes']==0
perf=[e for e in ev if e['event']=='measure'];quality=[e for e in ev if e['event']=='quality'];diag=[e for e in ev if e['event']=='diagnostic']
for e in perf+quality+diag:
 c=fc[e['case']];m=e['metrics'];assert m['prompt_tokens']==len(c['tokens']) and m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8
 assert m['prompt_tokens']+m['generated_tokens']-1<=1280 and m['true_tail8_calls']==0
 assert m['generated_tokens']==len(m['tokens'])
 assert all(math.isfinite(m[k]) and m[k]>=0 for k in ['prefill_seconds','scalar_prompt_seconds','ttft_seconds','total_seconds'])
 if e['event']!='diagnostic':
  p=e['prefix_prepare_metrics'];assert p['reused_tokens']==0 and p['prompt_tokens']==c['static_tokens'] and p['new_tokens']==c['static_tokens'] and not p['diagnostic']
  assert not m['diagnostic'] and len(e['token_events'])==m['generated_tokens']
  assert [t['token'] for t in e['token_events']]==m['tokens']
  assert e['token_events'][0]['elapsed']==m['ttft_seconds']
  assert all(b['elapsed']>a['elapsed'] for a,b in zip(e['token_events'],e['token_events'][1:]))
 if e['event']=='measure':
  assert m['fixed_feedback'] and m['generated_tokens']==32 and e['feedback_tokens']==cfg['feedback_tokens']
  assert abs(e['dynamic_prefill_seconds']-m['prefill_seconds']-m['scalar_prompt_seconds'])<1e-10
  assert sum(z['intervals'] for z in e['decode_segments'])==31
  assert abs(sum(z['seconds'] for z in e['decode_segments'])-(e['token_events'][-1]['elapsed']-e['token_events'][0]['elapsed']))<1e-8
 if e['event']=='quality':assert not m['fixed_feedback'] and m['generated_tokens']<=32 and e['input']==c['input']
 if e['event']=='diagnostic':assert m['diagnostic'] and m['fixed_feedback'] and m['generated_tokens']==4 and e['timing_eligible'] is False
summary={'audit_pass':True,'counts':dict(counts),'source_checks_pass':True,'input_checks_pass':True,'only_changed_graphs':changed,'cells':[],'matched_drift':{},'diagnostics':[],'quality':[]}
for s in (64,128):
 for d in (64,128,192,256):
  cell={'static':s,'dynamic_body':d,'actual_new':d+8,'variants':{}}
  for n in names:
   r=[e for e in perf if e['variant']==n and e['static_tokens']==s and e['body_tokens']==d];assert len(r)==3 and len({e['case'] for e in r})==3
   cell['variants'][n]={k:stat(v) for k,v in {
    'prefill_s':[e['metrics']['prefill_seconds']+e['metrics']['scalar_prompt_seconds'] for e in r],
    'ttft_s':[e['metrics']['ttft_seconds'] for e in r],
    'decode_tps':[31/(e['metrics']['total_seconds']-e['metrics']['ttft_seconds']) for e in r],
    'setup_s':[e['metrics']['setup_seconds'] for e in r],
    'first_head_s':[e['metrics']['first_head_seconds'] for e in r]}.items()}
  summary['cells'].append(cell)
base={e['case']:e for e in perf if e['variant']==names[0]}
for n in names[1:]:
 rr=[e for e in perf if e['variant']==n];tt=[e['metrics']['ttft_seconds']/base[e['case']]['metrics']['ttft_seconds'] for e in rr];de=[(base[e['case']]['metrics']['total_seconds']-base[e['case']]['metrics']['ttft_seconds'])/(e['metrics']['total_seconds']-e['metrics']['ttft_seconds']) for e in rr]
 summary['matched_drift'][n]={'ttft_ratio':stat(tt),'decode_ratio':stat(de),'fixed_prediction_full_exact':sum(e['metrics']['tokens']==base[e['case']]['metrics']['tokens'] for e in rr),'fixed_first_exact':sum(e['metrics']['tokens'][0]==base[e['case']]['metrics']['tokens'][0] for e in rr),'n':24}
db={}
for e in diag:
 path=D/e['logits_file'];raw=path.read_bytes();assert len(raw)==2426784==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
 x=np.frombuffer(raw,dtype='<f4').reshape(4,151674);assert np.isfinite(x).all();assert x.argmax(1).tolist()==e['metrics']['tokens']
 db[(e['variant'],e['case'])]=x
for n in names[1:]:
 for case in [e['case'] for e in diag if e['variant']==names[0]]:
  a=db[(names[0],case)];b=db[(n,case)];delta=b.astype('f8')-a;out={'variant':n,'case':case,'bitwise':a.tobytes()==b.tobytes(),'rel_l2':float(np.linalg.norm(delta)/np.linalg.norm(a.astype('f8'))),'max_abs':float(np.max(np.abs(delta))),'steps':[]}
  for step in range(4):
   ai=int(a[step].argmax());bi=int(b[step].argmax());aa=np.sort(a[step]);bb=np.sort(b[step]);out['steps'].append({'step':step,'baseline_top1':ai,'candidate_top1':bi,'same_top1':ai==bi,'baseline_top1_margin':float(aa[-1]-aa[-2]),'candidate_top1_margin':float(bb[-1]-bb[-2]),'baseline_top1_tie_count':int(np.sum(a[step]==a[step,ai])),'candidate_top1_tie_count':int(np.sum(b[step]==b[step,bi]))})
  summary['diagnostics'].append(out)
old=J(T/'fullconv-quality/results/fullconv-long-quality-r1/INDEPENDENT_SEMANTIC_BLINDED.json');review={(x['input_tokens_sha256'],x['reply']):x for x in old['rows']}
qb={e['case']:e for e in quality if e['variant']==names[0]}
for e in quality:
 c=fq[e['case']];key=(c['tokens_sha256'],e['reply']);prev=review.get(key);row={'variant':e['variant'],'case':e['case'],'reply':e['reply'],'reply_sha256':hashlib.sha256(e['reply'].encode()).hexdigest(),'input_sha256':c['tokens_sha256'],'generated_tokens':e['metrics']['generated_tokens'],'baseline_exact':e['metrics']['tokens']==qb[e['case']]['metrics']['tokens'],'first_exact':e['metrics']['tokens'][0]==qb[e['case']]['metrics']['tokens'][0],'gold':c['gold'],'top3':e['top3'],'manual_review':None}
 if prev:row['manual_review']={'semantic_class':prev['semantic_class'],'explicit_conflict':prev['explicit_conflict'],'reason':prev['reason'],'method':'Reuse prior manual rubric only after exact input-token hash and reply-byte equality; no inference from parity.'}
 summary['quality'].append(row)
for n in names:
 rr=[x for x in summary['quality'] if x['variant']==n];assert len(rr)==24 and len({x['case'] for x in rr})==24
summary['quality_manual_pending']=[{'variant':x['variant'],'case':x['case'],'reply':x['reply'],'gold':x['gold']} for x in summary['quality'] if x['manual_review'] is None]
# Whole-system / process interval statistics avoid assuming clock tick frequency.
def cpucount(r):
 a=list(map(int,r['cpu'][0].split()[1:9]));return sum(a),sum(a)-a[3]-a[4]
def pcputicks(v):
 a=v['stat'].rsplit(')',1)[1].split();return int(a[11])+int(a[12])
def resource_report(samples,intervals):
 assert samples
 o={'samples':len(samples),'temperature_C':stat([r['temp_mC']/1000 for r in samples]),'MemAvailable_MiB':stat([r['meminfo']['MemAvailable']/1024 for r in samples]),'RSS_sum_MiB':stat([sum(p.get('VmRSS',0) for p in r['processes'].values())/1024 for r in samples]),'PSS_sum_MiB':stat([sum(p.get('Pss',0) for p in r['processes'].values())/1024 for r in samples]),'SwapUsed_MiB':stat([(r['meminfo']['SwapTotal']-r['meminfo']['SwapFree'])/1024 for r in samples]),'load1':stat([float(r['loadavg'].split()[0]) for r in samples]),'freq_khz':{c:stat([r['freq_khz'][c] for r in samples if c in r['freq_khz']]) for c in samples[0]['freq_khz']},'max_audio_queue':max(r['frontend']['queue'] for r in samples)}
 total=busy=pt=0;swpin=swpout=0;sec=0;frames=overruns=0
 for a,b in intervals:
  at,ab=cpucount(a);bt,bb=cpucount(b);dt=bt-at
  if dt<=0:continue
  total+=dt;busy+=bb-ab;sec+=b['at']-a['at'];swpin+=b['vmstat']['pswpin']-a['vmstat']['pswpin'];swpout+=b['vmstat']['pswpout']-a['vmstat']['pswpout']
  for pid in a['processes'].keys()&b['processes'].keys():pt+=max(0,pcputicks(b['processes'][pid])-pcputicks(a['processes'][pid]))
  frames+=b['frontend']['frames']-a['frontend']['frames'];overruns+=b['frontend']['overruns']-a['frontend']['overruns']
 o.update({'interval_seconds':sec,'system_percent_whole8_weighted':100*busy/total if total else None,'sampled_process_cores_weighted':8*pt/total if total else None,'swapin_pages':swpin,'swapout_pages':swpout,'audio_interval_frames':frames,'audio_interval_overruns':overruns,'scope':'phase slices include system-cache rebuilds; CPU process sum tracks harness and resident, driver separately'})
 return o
intervals=list(zip(rs,rs[1:]));summary['resources_global']=resource_report(rs,intervals);summary['resource_phases']={}
for n in names:
 for ph in ['performance','quality','diagnostic']:
  sample=[r for r in rs if r['variant']==n and r['phase']==ph];ivs=[(a,b) for a,b in intervals if a['variant']==b['variant']==n and a['phase']==b['phase']==ph]
  if sample:summary['resource_phases'][n+'/'+ph]=resource_report(sample,ivs)
summary['audio_final']=meta['audio'];summary['driver_ready']={e['variant']:e['ready']['driver_bytes'] for e in loads};summary['driver_closed']={e['variant']:e['metrics']['driver_bytes'] for e in closes};summary['whole_run_s']=ev[-1]['at']-ev[0]['at']
summary['actual_data_hashes']={n:sha(D/n) for n in ['events.jsonl','resources.jsonl','meta.json','fixtures.json','config.json','provenance.json']}
summary['script_sha256']=sha(pathlib.Path(__file__))
summary['scope_limits']=['C1280 only, no true-tail8; one fused2 prefill graph replaced, not a new global precision format.','3 cost cases per cell, one repeat, not stable P95 or long-duration reliability.','24 natural presentations are 12 scene bodies x2system and reused development data. Historical independent_holdout fixture flag is stale for this experiment.','Audio prerecorded independent stream; ASR output does not drive fixed LLM prompts; no visual encoder, microphone or motor.','Performance fixed feedback32; natural max32/EOS; diagnostics4 forced steps excluded from speed.','System cache rebuilt each request before measured hot request; prep is excluded from TTFT but included in whole-run resources.','Token/logit parity is numerical reproducibility, not semantic correctness or action safety.']
(D/'RECOMPUTED_NUMERIC_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':dict(counts),'parity':summary['matched_drift'],'diag_exact':sum(x['bitwise'] for x in summary['diagnostics']),'diag_top1_matches':sum(y['same_top1'] for x in summary['diagnostics'] for y in x['steps']),'manual_pending':summary['quality_manual_pending'],'max_temp':summary['resources_global']['temperature_C']['max'],'driver':summary['driver_closed']},ensure_ascii=False))
