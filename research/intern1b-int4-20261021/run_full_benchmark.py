from pathlib import Path
import json,time,sys,subprocess,fcntl,shlex,hashlib,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;root=Path('/opt/opi-lab/intern1b-int4-20261021');d='/opt/opi-vlm/intern1b-int4-side-20261021'
def state(phase,**kw):
 value={'phase':phase,'at':time.time(),**kw};(r/'status.json').write_text(json.dumps(value,indent=2));print(value,flush=True)
state('waiting_for_compile')
for _ in range(180):
 p=r/'compile-progress.json'
 if p.exists():
  report=json.loads(p.read_text())
  if report['status']=='failed':raise RuntimeError('compile failed')
  if report['status']=='complete':break
 time.sleep(5)
else:raise RuntimeError('compile deadline exceeded')
try:
 records=report['records'];assert len(records)==28
 stage=root/'stage';stage.mkdir(exist_ok=True);inventory={}
 for rec in records:
  p=root/rec['kind']/('l'+str(rec['layer']));src=Path(rec['source']);metap='export_nbg_unify/nbg_meta.json'
  left=json.loads((p/metap).read_text());right=json.loads((src/metap).read_text());assert left==right
  assert all(list(left[x])==list(right[x]) for x in ['Inputs','Outputs'])
  name=('prefill-' if rec['kind']=='prefill' else '')+'layer'+str(rec['layer'])+'.nb';target=stage/name
  if not target.exists():target.symlink_to(p/'export_nbg_unify/network_binary.nb')
  inventory[name]={'sha256':rec['binary_sha256'],'bytes':rec['binary_bytes']}
 (r/'deployment-manifest.json').write_text(json.dumps(inventory,indent=2))
 lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state('deploying',bytes=sum(x['bytes'] for x in inventory.values()))
 cmd='mkdir -p '+d+'/w4 '+d+'/results'
 b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),check=True,timeout=30)
 for n in ['board_benchmark.py','resident_worker.py','inputs.json','deployment-manifest.json']:b.scp(r/n,b.HOST+':'+d+'/'+n,check=True,timeout=60)
 subprocess.run(['sshpass','-e','rsync','-rtLz','--compress-level=1','--partial','--append-verify','-e','ssh '+' '.join(b.COMMON),str(stage)+'/',b.HOST+':'+d+'/w4/'],env=b.env(),check=True,timeout=1200)
 check="from pathlib import Path;import json,hashlib; r=Path('"+d+"');m=json.loads((r/'deployment-manifest.json').read_text());assert all(hashlib.sha256((r/'w4'/n).read_bytes()).hexdigest()==v['sha256'] for n,v in m.items());p=r/'w4/embeddings.f32';p.symlink_to('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018/embeddings.f32') if not p.exists() else None;print('all28 graphs SHA verified')"
 b.ssh('flock -n /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(check),check=True,timeout=180)
 state('benchmarking')
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock timeout --kill-after=15s 1800s /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+d+'/board_benchmark.py',timeout=1850)
 state('collecting',benchmark_exit=z.returncode)
 subprocess.run(['sshpass','-e','rsync','-rtz','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+d+'/results/',str(r/'results')+'/'],env=b.env(),check=True,timeout=300)
 state('returned',benchmark_exit=z.returncode)
except BaseException as e:state('error',error=repr(e),trace=traceback.format_exc());raise
