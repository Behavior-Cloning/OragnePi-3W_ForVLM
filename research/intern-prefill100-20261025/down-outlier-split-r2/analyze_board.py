from pathlib import Path
import json,hashlib,math,statistics
import numpy as np
R=Path(__file__).resolve().parent;D=R/'results';cfg=json.loads((D/'config.json').read_text());meta=json.loads((D/'meta.json').read_text());control=json.loads((R/'controller.json').read_text())
assert meta['status']=='complete' and not meta['monitor_errors'] and control['returncode']==0
E=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];results={x['case']:x for x in E if x['event']=='result'};assert len(results)==48
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):return {'n':len(v),'p50':float(np.median(v)),'p95_nearest_rank':float(sorted(v)[math.ceil(.95*len(v))-1]),'min':float(min(v)),'max':float(max(v))}
def parts(a,valid):
 z={'hidden':a[:16*896].reshape(16,896)[:valid]};o=16*896
 for n in ['K2','K3','V2','V3']:z[n]=a[o:o+16*128].reshape(2,16,64)[:,:valid,:];o+=16*128
 return z
def diff(a,b):
 x=a.astype('float64');y=b.astype('float64');return {'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),'max_abs':float(np.max(np.abs(x-y))),'mismatches':int(np.count_nonzero(x!=y)),'finite':bool(np.isfinite(x).all())}
rows=[];arrays={}
for f in cfg['sequence']:
 e=results[f['run_id']];m=e['metrics'];path=D/(f['run_id']+'.f32');raw=path.read_bytes();assert len(raw)==f['expected_output_bytes'] and hashlib.sha256(raw).hexdigest()==e['output_sha256']
 a=np.frombuffer(raw,'<f4');assert np.isfinite(a).all();arrays[f['run_id']]=a
 assert m['released_bytes']==0 and m['repeats']==10 and m['warmup_repeats']==3 and len(m['api_seconds'])==10 and not m['npd_enabled'] and not m['io_profile_enabled'] and m['mismatched_elements']==0 and m['differing_repeats']==0
 rows.append({'key':f['key'],'variant':f['variant'],'case':f['case'],'position':f['position'],'valid_rows':f['valid_rows'],'api_ms':stats([x*1000 for x in m['api_seconds']]),'api_ms_values':[x*1000 for x in m['api_seconds']],'sha256':e['output_sha256'],'driver_zero':True})
for row in rows:
 a=arrays[row['key']+'-'+row['variant']];b=arrays[row['key']+'-original'];aa=parts(a,row['valid_rows']);bb=parts(b,row['valid_rows']);row['error_vs_same_input_original']={k:diff(aa[k],bb[k]) for k in aa}
 if row['position']==0:row['first_token_hidden_error']=diff(aa['hidden'][0],bb['hidden'][0])
 if row['variant'] in ['original','original-repeat']:
  assert np.array_equal(a.view('uint8'),b.view('uint8'))
  f=next(f for f in cfg['fixtures'] if f['key']==row['key']);assert hashlib.sha256(a.tobytes()).hexdigest()==f['original_output_sha256']
variants=['original','row-scale','outlier8','original-repeat'];summary={}
for v in variants:
 selected=[x for x in rows if x['variant']==v];summary[v]={'all_windows_api_ms':stats([t for r in selected for t in r['api_ms_values']]),'first_token_hidden_errors':[x['first_token_hidden_error'] for x in selected if x['position']==0],'dynamic_hidden_errors':[x['error_vs_same_input_original']['hidden']['relative_l2'] for x in selected if x['position']>=(64 if x['case'].startswith('s64-') else 128)]}
res=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()]
def vm(x):return {line.split()[0]:int(line.split()[1]) for line in x['vmstat'].splitlines()}
a=vm(res[0]);b=vm(res[-1]);resource={'samples':len(res),'span_seconds':res[-1]['at']-res[0]['at'],'temperature_peak_C':max(max(x['temperatures_mC'].values()) for x in res)/1000,'MemAvailable_min_MiB':min(int(x['memory']['MemAvailable'].split()[0]) for x in res)/1024,'swapin_pages':b['pswpin']-a['pswpin'],'swapout_pages':b['pswpout']-a['pswpout']}
out={'status':'local-screen-complete-not-whole-model-qualified','scope':'L2/L3 M16 C1280, 12 windows from two existing prompts, 4 variants A/B/C/A, 3 warmup and10 timed calls each. Local API excludes load, conversion, whole model, head, audio/RAG.','calibration':'k8 picked using s64-d64-0 capture only; s128-d256-0 heldout from selection but reused historical prompt','summary':summary,'rows':rows,'resources':resource,'closed_driver0':48,'sources_sha256':{str(f.relative_to(R)):sha(f) for f in [Path(__file__),R/'STATIC_AUDIT.json',R/'config.json',D/'events.jsonl',D/'resources.jsonl',D/'meta.json']}}
(R/'BOARD_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'summary':summary,'resources':resource},indent=2))
