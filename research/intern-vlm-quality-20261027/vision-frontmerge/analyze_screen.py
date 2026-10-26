from pathlib import Path
import json,statistics,re,hashlib,numpy as np
D=Path(__file__).resolve().parent;P=D/'board-r4-results/results';s=json.loads((P/'summary.json').read_text());assert s['state']=='completed'
for n,h in s['sha256'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h
rows=[json.loads(x) for x in (P/'resources.jsonl').read_text().splitlines()]
def val(s,k):
 m=re.search(r'^'+k+r':\s+(\d+)',s,re.M);return int(m.group(1)) if m else None
by_pid={}
for x in rows:
 p=val(x.get('process_status',''),'Pid')
 if p is not None:by_pid.setdefault(p,[]).append(x)
assert len(by_pid)==6
ph=[]
for phase,(pid,rs) in zip(s['phases'],by_pid.items()):
 ev=[x for x in phase['events'] if x['event']=='encoded'];warm=[x['encoder_projector_ms'] for x in ev[1:]]
 ph.append({'phase':phase['phase'],'vit_patch_tokens':ev[0]['vit_patch_tokens'],'image_tokens':ev[0]['image_tokens'],'process_load_ms':phase['events'][0]['load_ms'],'warm_n':len(warm),'warm_P50_ms':statistics.median(warm),'warm_min_ms':min(warm),'warm_max_ms':max(warm),'max_RSS_MiB':max(val(x['process_status'],'VmRSS') or 0 for x in rs)/1024,'max_temp_C':max(int(v)/1000 for x in rs for v in x['temperatures'].values()),'pid':pid})
checks={}
for i in [0,1,2]:checks['native_repeat_'+str(i)]=s['sha256']['native-before-'+str(i)+'.f32']==s['sha256']['native-after-'+str(i)+'.f32']
for i in [1,2]:
 a=np.fromfile(P/('native-before-'+str(i)+'.f32'),dtype='<f4').reshape(16,16,896);expected=a.reshape(8,2,8,2,896).mean((1,3));got=np.fromfile(P/('late-'+str(i)+'.f32'),dtype='<f4').reshape(8,8,896)
 checks['late_pool_max_abs_'+str(i)]=float(np.max(np.abs(expected-got)))
 assert np.allclose(expected,got,atol=2e-6,rtol=2e-6)
assert all(v for k,v in checks.items() if k.startswith('native_repeat'))
ans=[json.loads(x) for x in (D/'vqa-answers.jsonl').read_text().splitlines()];assert sum(x.get('event')=='answer' for x in ans)==32 and not any(x.get('event')=='error' for x in ans)
# Manual evaluation against frozen references, NOT an independent blinded benchmark.
scores={'native-before':[1,1,1,1,1,1],'front':[1,0,0,0,0,0],'late':[1,1,1,1,1,1],'native224':[1,0,1,1,1,1],'keep112':[0,0,0,0,0,0]}
for p in ph:
 if p['phase'] in scores:p['correct_core_fact_out_of_6']=sum(scores[p['phase']])
audit={'phases':ph,'numerical_checks':checks,'resources':json.loads((P/'resource-summary.json').read_text()),'quality_scope':'2 real example images / 6 easy development questions; WSL CPU Q8 language reference; no end-to-end board NPU VQA yet; not held-out validation','manual_scores_in_frozen_case_order':scores,'quality_qualified':False,'default_changed':False,'vision_quantization':'F16 linear weights, F32 metadata/norm/etc; no Q8 vision test yet','source_binary':json.loads((D/'build-r3-status.json').read_text()),'VQA_all_same_system_tokens':sorted(set(x['system_total_tokens'] for x in ans if x.get('event')=='answer'))}
(D/'SCREEN_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps({'phases':ph,'checks':checks},ensure_ascii=False))
