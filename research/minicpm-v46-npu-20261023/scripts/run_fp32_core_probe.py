from pathlib import Path
import sys,json,shlex,fcntl,numpy as np
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);d='/opt/opi-vlm/minicpm-v46-npu-20261023'
kind='core-fp32';src=bld/'matmul-b1';nb=bld/'core-fp32'/'export_nbg_unify/network_binary.nb'
for p,n in [(src/'input.f32',kind+'-input.f32'),(nb,kind+'.nb')]:b.scp(p,b.HOST+':'+d+'/'+n,check=True,timeout=120)
cmd='cd '+d+' && env OPI_PROBE_WARMUP=5 timeout --kill-after=5s 60s taskset -c 6,7 ./probe '+kind+'.nb '+kind+'-input.f32 '+kind+'-output.f32 30 '+kind+'-timing.json'
z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=70);print(z.stdout,z.stderr,flush=True);assert z.returncode==0
for n in [kind+'-output.f32',kind+'-timing.json']:b.scp(b.HOST+':'+d+'/'+n,r/'results'/n,check=True,timeout=30)
a=np.fromfile(r/'results'/(kind+'-output.f32'),np.float32);ref=np.fromfile(src/'reference.f32',np.float32);out=[];offset=0
for n,count in [('out',2048),('state',262144)]:
 x=a[offset:offset+count];y=ref[offset:offset+count];offset+=count;out.append({'name':n,'max_abs':float(abs(x-y).max()),'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'finite':bool(np.isfinite(x).all())})
row={'timing':json.loads((r/'results'/(kind+'-timing.json')).read_text()),'errors':out};(r/'results'/(kind+'-probe.json')).write_text(json.dumps(row,indent=2));print(row,flush=True)
