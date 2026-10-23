from pathlib import Path
import subprocess,sys,fcntl,shlex,json,hashlib
import numpy as np
r=Path(__file__).resolve().parents[1];build=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d='/opt/opi-vlm/minicpm-v46-npu-20261023'
def run(cmd,timeout=45):
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout)
 print(z.stdout,z.stderr,flush=True)
 if z.returncode:raise RuntimeError('board return '+str(z.returncode))
 return z
run('mkdir -p '+d)
b.scp(r/'scripts/board_cleanup.py',b.HOST+':'+d+'/board_cleanup.py',check=True,timeout=30);run('python3 '+d+'/board_cleanup.py');b.scp(b.HOST+':'+d+'/cleanup.json',r/'results/board-cleanup.json',check=True,timeout=30)
for n in ['tensor_probe.c','full_model.c','board_probe.c','greedy_i16.h','batch_prefill.h']:b.scp(r.parents[1]/'npu/kv-cache'/n,b.HOST+':'+d+'/'+n,check=True,timeout=30)
run('cd '+d+' && cc -O2 -Wall -Wextra -Werror -DFUSED_LAYERS=2 tensor_probe.c -lNBGlinker -lVIPhal -lm -o probe')
summary=[]
for kind in ['reduce-b1','matmul-b1']:
 src=build/kind;nb=build/(kind+'-fp16')/'export_nbg_unify/network_binary.nb'
 for path,name in [(src/'input.f32',kind+'-input.f32'),(nb,kind+'.nb')]:b.scp(path,b.HOST+':'+d+'/'+name,check=True,timeout=45)
 run('cd '+d+' && env OPI_PROBE_WARMUP=5 timeout --kill-after=5s 45s taskset -c 6,7 ./probe '+kind+'.nb '+kind+'-input.f32 '+kind+'-output.f32 50 '+kind+'-timing.json',55)
 for name in [kind+'-output.f32',kind+'-timing.json']:b.scp(b.HOST+':'+d+'/'+name,r/'results'/name,check=True,timeout=30)
 actual=np.fromfile(r/'results'/(kind+'-output.f32'),np.float32);ref=np.fromfile(src/'reference.f32',np.float32);metrics=[]
 for name,sl in [('out',slice(0,2048)),('state',slice(2048,None))]:
  a=actual[sl];v=ref[sl];metrics.append({'name':name,'relative_l2':float(np.linalg.norm(a-v)/np.linalg.norm(v)),'max_abs':float(abs(a-v).max()),'finite':bool(np.isfinite(a).all())})
 row={'kind':kind,'timing':json.loads((r/'results'/(kind+'-timing.json')).read_text()),'error':metrics,'nb_sha256':hashlib.sha256(nb.read_bytes()).hexdigest()};summary.append(row);(r/'results/core-probe-summary.json').write_text(json.dumps(summary,indent=2));print(row,flush=True)
