from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;D=R/'results';state=json.loads((R/'controller.json').read_text());assert state['status']=='complete' and state['returncode']==0
E=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];items=[x for x in E if x['event']=='result'];assert len(items)==10
arrays={};out={'scope':'L2/L3 two-layer attention joint scores/value/both;history128;no systemtext/template/head;no full model/speech/RAG','results':[],'comparisons':[]}
old=json.loads((R.parent/'m64-p448-screen-r8/ANALYSIS.json').read_text())['results']
for e in items:
 n=e['case'];q=e['query_tokens'];m=e['metrics'];p=D/(n+'.f32');b=p.read_bytes();assert len(b)==4*q*(896+4*128) and hashlib.sha256(b).hexdigest()==e['output_sha256'];a=np.frombuffer(b,'<f4');assert np.isfinite(a).all();arrays[n]=a
 assert m['released_bytes']==0 and len(m['api_seconds'])==30 and m['mismatched_elements']==0 and m['differing_repeats']==0 and not m['npd_enabled'] and not m['io_profile_enabled'] and m['resident_copies']==0
 out['results'].append({'variant':n,'query':q,'past':m['capacity'],'n':30,'p50_ms':float(np.median(m['api_seconds'])*1000),'p95_ms':float(sorted(m['api_seconds'])[28]*1000),'sha256':e['output_sha256'],'bytes':len(b),'finite':True,'driver0':True})
for name,q in [('m32-p448',32),('m64-p128',64)]:
 ref=arrays[name+'-before'];assert np.array_equal(ref.view('u1'),arrays[name+'-after'].view('u1'))
 assert next(x for x in out['results'] if x['variant']==name+'-before')['sha256']==next(x for x in old if x['variant']==f'l2-m{q}-before')['sha256']
 for mode in ['scores','value','both']:
  cand=arrays[name+'-'+mode];delta=cand.astype('f8')-ref.astype('f8');h=q*896;segments=[];start=0
  for sn,length in [('hidden',h),('b0_new_k',q*128),('b0_new_v',q*128),('b1_new_k',q*128),('b1_new_v',q*128)]:
   rr=ref[start:start+length].astype('f8');dd=delta[start:start+length];segments.append({'tensor':sn,'max_abs':float(np.max(np.abs(dd))),'relative_l2':float(np.linalg.norm(dd)/max(np.linalg.norm(rr),1e-30))});start+=length
  A=[x for x in out['results'] if x['variant'] in [name+'-before',name+'-after']];B=next(x for x in out['results'] if x['variant']==name+'-'+mode);faster=min(x['p50_ms'] for x in A)
  out['comparisons'].append({'shape':name,'mode':mode,'reference_same_bytes_as_archived_r8':True,'A_before_ms':A[0]['p50_ms'],'B_ms':B['p50_ms'],'A_after_ms':A[1]['p50_ms'],'saved_vs_faster_A_percent':100*(faster-B['p50_ms'])/faster,'same_bytes':bool(np.array_equal(ref.view('u1'),cand.view('u1'))),'mismatched_elements':int(np.count_nonzero(delta)),'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(ref.astype('f8'))),'segments':segments})
res=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()];out['peak_temp_C']=max(max(x['temperatures_mC'].values()) for x in res)/1000;out['minimum_available_MiB']=min(int(x['memory']['MemAvailable'].split()[0]) for x in res)/1024;out['resource_samples']=len(res)
out['evidence_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'config.json',R/'deployment-sha.json',D/'events.jsonl',D/'resources.jsonl',D/'meta.json']};(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
