"""Queue board memory checks behind NPU matrix via the same host/board locks."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
repo=Path(__file__).resolve().parents[2];source=Path(__file__).resolve().parent
out=repo/'results/memory-lab-20261018';remote='/opt/opi-vlm/memory-lab-20261018/v2'
with (out/'v2-state.json').open('w') as f:json.dump({'status':'waiting-for-board-lock'},f)
with b.hostlock():
 b.ssh('mkdir -p '+remote,check=True,timeout=30)
 files=[source/x for x in ['memory_lab.py','prompt_budget.py','test_memory_lab.py','benchmark_memory.py','evaluate_guarded.py']]
 hashes=[]
 for p in files:
  b.scp(p,'root@192.168.10.11:'+remote+'/'+p.name,check=True,timeout=30);hashes.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
 (out/'v2-source.sha256').write_text('\n'.join(hashes)+'\n')
 b.ssh('cd '+remote+' && sha256sum -c -',input='\n'.join(hashes)+'\n',text=True,check=True,timeout=30)
 command='cd '+remote+' && python3 -m unittest -v test_memory_lab && python3 evaluate_guarded.py > guarded-probes.json'
 for n in [500,5000]:
  for strategy in ['profile_fts','profile_fts_guarded']:
   command+=' && taskset -c 0,1 python3 benchmark_memory.py --out results-'+strategy+'-'+str(n)+' --strategy '+strategy+' --count '+str(n)
 with (out/'v2-run.log').open('x') as log:
  r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command),stdout=log,stderr=subprocess.STDOUT,timeout=600)
 assert r.returncode==0
 results=[]
 for n in [500,5000]:
  for strategy in ['profile_fts','profile_fts_guarded']:
   target=out/('v2-'+strategy+'-'+str(n)+'.json')
   b.scp('root@192.168.10.11:'+remote+'/results-'+strategy+'-'+str(n)+'/result.json',target,check=True,timeout=30)
   results.append(json.loads(target.read_text()))
 b.scp('root@192.168.10.11:'+remote+'/guarded-probes.json',out/'v2-guarded-probes.json',check=True,timeout=30)
 (out/'v2-summary.json').write_text(json.dumps({'status':'complete','results':results},ensure_ascii=False,indent=2)+'\n')
 (out/'v2-state.json').write_text(json.dumps({'status':'complete'})+'\n')
 for r in results:print(json.dumps({k:v for k,v in r.items() if k not in ['trials','quality_by_category']},ensure_ascii=False),flush=True)
