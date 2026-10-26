from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;D=R/'results';state=json.loads((R/'controller.json').read_text());assert state['status']=='complete' and state['returncode']==0
E=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];items=[x for x in E if x['event']=='result'];assert len(items)==10
arrays={};out={'scope':'L0/L1 and L2/L3;canonical history128;M64P128 vsP448;M32P448 reference;pureAPI only;no full model or speech','results':[]}
for e in items:
 n=e['case'];q=e['query_tokens'];m=e['metrics'];p=D/(n+'.f32');b=p.read_bytes();assert len(b)==4*q*(896+4*128) and hashlib.sha256(b).hexdigest()==e['output_sha256'];a=np.frombuffer(b,'<f4');assert np.isfinite(a).all();arrays[n]=a
 assert m['released_bytes']==0 and len(m['api_seconds'])==30 and m['mismatched_elements']==0 and m['differing_repeats']==0 and not m['npd_enabled']
 out['results'].append({'variant':n,'layer':e['first_layer'],'query':q,'past':m['capacity'],'n':30,'p50_ms':float(np.median(m['api_seconds'])*1000),'p95_ms':float(sorted(m['api_seconds'])[28]*1000),'sha256':e['output_sha256'],'bytes':len(b),'finite':True,'driver0':True})
for layer in [0,2]:
 for q in [32,64]:assert np.array_equal(arrays[f'l{layer}-m{q}-before'].view('u1'),arrays[f'l{layer}-m{q}-after'].view('u1'))
for r in out['results']:
 a=arrays[r['variant']];ref=arrays[f"l{r['layer']}-m{r['query']}-before"];h=896*r['query'];r['same_bytes']=bool(np.array_equal(a.view('u1'),ref.view('u1')));r['relative_l2']=float(np.linalg.norm(a.astype('f8')-ref)/np.linalg.norm(ref.astype('f8')));r['hidden_relative_l2']=float(np.linalg.norm(a[:h].astype('f8')-ref[:h])/np.linalg.norm(ref[:h].astype('f8')));r['per_row_api_ms']=r['p50_ms']/r['query']
assert next(x for x in out['results'] if x['variant']=='l0-m64-before')['sha256']=='a333a5d51a6f45882b0b7bbb774b56b269bd1674d342c76ca2a32c2370dc3771'
out['old_probe_control_same_bytes']=True
out['throughput_reference']=[]
for layer in [0,2]:
 a=min(x['p50_ms'] for x in out['results'] if x['layer']==layer and x['query']==32);b=next(x['p50_ms'] for x in out['results'] if x['variant']==f'l{layer}-m64-p448');out['throughput_reference'].append({'layer':layer,'m32_faster_A_ms':a,'m64_p448_ms':b,'per_row_speedup':2*a/b,'caveat':'different query lengths;not two sequential M32 calls with updatedKV;not full-model speedup'})
res=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()];out['peak_temp_C']=max(max(x['temperatures_mC'].values()) for x in res)/1000;out['minimum_available_MiB']=min(int(x['memory']['MemAvailable'].split()[0]) for x in res)/1024
out['evidence_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'config.json',R/'deployment-sha.json',D/'events.jsonl',D/'resources.jsonl',D/'meta.json']};(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
