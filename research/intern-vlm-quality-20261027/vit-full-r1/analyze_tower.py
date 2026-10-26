from pathlib import Path
import json,numpy as np,statistics as st,re
P=Path(__file__).resolve().parent;O=P/'tower-r1/results'
ev=[json.loads(l) for l in (O/'events.jsonl').read_text().splitlines()];assert ev[0]['layers']==26 and ev[-1]['event']=='closed' and ev[-1]['driver_bytes']==0
status=json.loads((O/'status.json').read_text());assert status['returncode']==0 and not status['monitor_errors']
rows=[x for x in ev if x['event']=='result' and x['run']>=0];assert len(rows)==6
errors=[]
for i in [1,2]:
 got=np.fromfile(O/f'run{i-1}-output.f32',np.float32).astype('float64');ref=np.fromfile(P/f's224/embedding{i}.f32',np.float32).astype('float64');assert got.size==ref.size==57344 and np.isfinite(got).all()
 for run in [i+1,i+3]:assert np.array_equal(got,np.fromfile(O/f'run{run}-output.f32',np.float32).astype('float64'))
 errors.append({'image':i,'relative_l2':float(np.linalg.norm(ref-got)/np.linalg.norm(ref)),'cosine':float(ref@got/(np.linalg.norm(ref)*np.linalg.norm(got)))})
res=[json.loads(l) for l in (O/'resources.jsonl').read_text().splitlines()]
audit={'scope':'normalized224x224->NPU stem+24blocks+projector->64x896; excludes JPEG/normalization/language','ready':ev[0],'closed':ev[-1],'tower_p50':st.median(x['stack_seconds'] for x in rows),'tower_min':min(x['stack_seconds'] for x in rows),'tower_max':max(x['stack_seconds'] for x in rows),'api_sum_p50':st.median(sum(x['api_seconds']) for x in rows),'host_p50':st.median(x['host_copy_sync_seconds'] for x in rows),'errors':errors,'max_temp_mC':max(max(x['temperatures'].values()) for x in res),'min_available_kib':min(int(re.search(r'MemAvailable:\s+(\d+)',x['meminfo']).group(1)) for x in res)}
(P/'tower-r1/AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
