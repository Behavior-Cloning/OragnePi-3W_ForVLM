from pathlib import Path
import json,sys,fcntl,shlex,hashlib,subprocess
import numpy as np
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;out=r/'probe';out.mkdir(exist_ok=True)
w4=Path('/opt/opi-lab/intern1b-int4-20261021/decode/l0');w8=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-cap1-headmse/l0')
meta=json.loads((w4/'export_nbg_unify/nbg_meta.json').read_text());assert meta==json.loads((w8/'export_nbg_unify/nbg_meta.json').read_text())
rng=np.random.default_rng(20261021);values=[]
for n,t in meta['Inputs'].items():
 v=np.zeros(t['shape'],np.float32)
 if 'hidden' in n:v=rng.normal(0,.02,t['shape']).astype(np.float32)
 if 'mask' in n:v.fill(-10000);v.reshape(-1)[-1]=0
 if 'rope_cos' in n:v.fill(1)
 values.append(v.reshape(-1))
np.concatenate(values).tofile(out/'input.f32')
(out/'fixture.json').write_text(json.dumps({'purpose':'Execution probe with fixed synthetic hidden and zero KV; not language quality.','input_sha256':hashlib.sha256((out/'input.f32').read_bytes()).hexdigest(),'metadata':meta},indent=2))
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d='/opt/opi-vlm/intern1b-int4-side-20261021';b.ssh('flock -n /run/opi-npu-kv-benchmark.lock mkdir -p '+d,check=True)
repo=r.parents[1]/'npu/kv-cache'
for n in ['tensor_probe.c','full_model.c','board_probe.c','greedy_i16.h','batch_prefill.h']:b.scp(repo/n,b.HOST+':'+d+'/'+n,check=True,timeout=60)
b.scp(out/'input.f32',b.HOST+':'+d+'/input.f32',check=True,timeout=60)
b.scp(w4/'export_nbg_unify/network_binary.nb',b.HOST+':'+d+'/w4.nb',check=True,timeout=180)
cmd='cd '+d+' && cc -O2 -Wall -Wextra -Werror -DFUSED_LAYERS=2 tensor_probe.c -lNBGlinker -lVIPhal -lm -o probe && ln -s /opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018/layer0.nb w8.nb'
b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),check=True,timeout=60)
for label in ['w8','w4','w4','w8']:
 i=len(list(out.glob('*-timing.json')));name=str(i)+'-'+label
 cmd='cd '+d+' && env OPI_PROBE_WARMUP=5 timeout --kill-after=5s 60s taskset -c 6,7 ./probe '+label+'.nb input.f32 '+name+'-output.f32 50 '+name+'-timing.json'
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=70)
 (out/(name+'.log')).write_text(z.stdout+z.stderr);print(label,z.returncode,flush=True)
 if z.returncode:raise RuntimeError(z.stdout[-2000:]+z.stderr[-2000:])
 for f in [name+'-timing.json',name+'-output.f32']:b.scp(b.HOST+':'+d+'/'+f,out/f,check=True,timeout=30)
 print((out/(name+'-timing.json')).read_text(),flush=True)
