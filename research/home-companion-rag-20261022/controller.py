from pathlib import Path
import sys,subprocess,shlex,json,time,traceback,hashlib
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/f/Files/OPi-Lab/experiments/board-rag-atoms-20261021/larger256-20261022-v1')
DEST='/opt/opi-vlm/board-rag-atoms-20261021/home-companion-20261022-v1'
PY='/opt/opi-vlm/jev-lab-20261020/venv/bin/python'
def status(phase,**kw):
 obj=dict(at=time.time(),phase=phase,**kw);(ROOT/'status.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj),flush=True)
def sync(src,dst,timeout=1800):
 subprocess.run(['sshpass','-e','rsync','-rt','--partial','--append-verify','-e','ssh '+' '.join(b.COMMON),str(src),str(dst)],env=b.env(),check=True,timeout=timeout)
keys=sys.argv[1:] or ['bge_small','gte_base','bge_base','bge_large','qwen06']
for key in keys:
 if (ROOT/'results'/key/'summary.json').exists():status('already_complete',model=key);continue
 status('waiting_shared_board_lock',model=key)
 with b.hostlock():
  try:
   status('deploying',model=key)
   b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock mkdir -p '+DEST+'/models',check=True,timeout=30)
   for n in ['board_bench.py','cards.json','cases.json']:
    b.scp(ROOT/n,b.HOST+':'+DEST+'/'+n,check=True,timeout=60)
   if key!='bge_small':sync(str(SOURCE/'models'/key),b.HOST+':'+DEST+'/models/')
   status('running',model=key)
   shell='set -o pipefail; timeout --kill-after=15s 2400s taskset -c 6,7 '+PY+' -B -u '+DEST+'/board_bench.py '+key+' 2>&1 | tee '+DEST+'/'+key+'-run.log'
   rc=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock bash -c '+shlex.quote(shell),timeout=2430).returncode
   (ROOT/'results'/key).mkdir(parents=True,exist_ok=True)
   sync(b.HOST+':'+DEST+'/results/'+key+'/',str(ROOT/'results'/key)+'/')
   b.scp(b.HOST+':'+DEST+'/'+key+'-run.log',ROOT/(key+'-run.log'),check=True,timeout=60)
   if rc:raise RuntimeError('board exit '+str(rc))
   summary=json.loads((ROOT/'results'/key/'summary.json').read_text());assert summary['complete']
   for name,h in summary['output_sha'].items():assert hashlib.sha256((ROOT/'results'/key/name).read_bytes()).hexdigest()==h
   if key!='bge_small':
    b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock rm -rf -- '+shlex.quote(DEST+'/models/'+key),check=True,timeout=60)
   status('model_complete',model=key)
  except Exception as e:
   (ROOT/(key+'-error.json')).write_text(json.dumps({'error':repr(e),'trace':traceback.format_exc()},indent=2));status('model_error',model=key,error=repr(e))
status('finished',models=keys)
