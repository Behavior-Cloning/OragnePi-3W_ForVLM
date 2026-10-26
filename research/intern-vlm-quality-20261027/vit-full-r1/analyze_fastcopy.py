from pathlib import Path
import json,statistics as st,hashlib,re,sys
P=Path(__file__).resolve().parent;O=P/(sys.argv[1] if len(sys.argv)>1 else 'fastcopy-r2');rows=[];hashes={}
for phase in ['base','fast','repeat']:
 d=O/phase/'results';ev=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()];assert ev[-1]['event']=='closed' and ev[-1]['driver_bytes']==0
 a=json.loads((d/'status.json').read_text());assert a['returncode']==0 and not a['monitor_errors']
 e=[x for x in ev if x['event']=='result' and x['run']>=0];assert len(e)==6
 hashes[phase]=[hashlib.sha256((d/f'run{i}-output.f32').read_bytes()).hexdigest() for i in range(6)]
 res=[json.loads(l) for l in (d/'resources.jsonl').read_text().splitlines()]
 rows.append({'phase':phase,'p50_s':st.median(x['stack_seconds'] for x in e),'min_s':min(x['stack_seconds'] for x in e),'max_s':max(x['stack_seconds'] for x in e),'api_s':st.median(sum(x['api_seconds']) for x in e),'host_s':st.median(x['host_copy_sync_seconds'] for x in e),'max_temp_mC':max(max(x['temperatures'].values()) for x in res),'min_available_kib':min(int(re.search(r'MemAvailable:\s+(\d+)',x['meminfo']).group(1)) for x in res)})
assert hashes['base']==hashes['fast']==hashes['repeat']
assert hashes['fast']==[hashlib.sha256((P/f'tower-r2/results/run{i}-output.f32').read_bytes()).hexdigest() for i in range(6)]
audit={'phases':rows,'byte_equal_all_outputs':True,'sha256':hashes,'selftest':(O/'selftest.txt').read_text(),'closed_driver_zero':True}
(O/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
