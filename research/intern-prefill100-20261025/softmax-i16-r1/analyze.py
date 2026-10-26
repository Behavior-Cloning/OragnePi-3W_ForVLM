from pathlib import Path
import json,hashlib,math,numpy as np
R=Path(__file__).resolve().parent;D=R/'results';C=json.loads((R/'controller.json').read_text());assert C['status']=='complete' and C['returncode']==0
E=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];rows=[x for x in E if x['event']=='result'];assert len(rows)==3
arrays={};out={'scope':'two-layer M64/past128 pure vip API; no full TTFT or audio/RAG','results':[]}
for row in rows:
 name=row['case'];m=row['metrics'];p=D/(name+'.f32');raw=p.read_bytes();assert len(raw)==4*64*(896+4*128) and hashlib.sha256(raw).hexdigest()==row['output_sha256'];a=np.frombuffer(raw,'<f4');assert np.isfinite(a).all();arrays[name]=a
 assert m['released_bytes']==0 and m['mismatched_elements']==0 and m['differing_repeats']==0 and len(m['api_seconds'])==30 and not m['npd_enabled']
 out['results'].append({'variant':name,'n':30,'p50_ms':float(np.median(m['api_seconds'])*1000),'p95_ms':float(sorted(m['api_seconds'])[28]*1000),'sha256':row['output_sha256'],'finite':True,'driver0':True,'bytes':len(raw)})
assert np.array_equal(arrays['before'].view('u1'),arrays['after'].view('u1'))
for row in out['results']:
 a=arrays[row['variant']];ref=arrays['before'];row['same_bytes']=bool(np.array_equal(a.view('u1'),ref.view('u1')));row['relative_l2']=float(np.linalg.norm(a.astype('f8')-ref)/np.linalg.norm(ref.astype('f8')))
 h=896*64;row['hidden_relative_l2']=float(np.linalg.norm(a[:h].astype('f8')-ref[:h])/np.linalg.norm(ref[:h].astype('f8')))
res=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()]
out['peak_temp_C']=max(max(x['temperatures_mC'].values()) for x in res)/1000
out['minimum_available_MiB']=min(int(x['memory']['MemAvailable'].split()[0]) for x in res)/1024
out['evidence_sha256']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'config.json',R/'deployment-sha.json',D/'events.jsonl',D/'resources.jsonl',D/'meta.json']}
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
