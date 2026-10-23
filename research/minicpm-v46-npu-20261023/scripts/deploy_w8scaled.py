from pathlib import Path
import sys,subprocess,json,shlex,hashlib,time,fcntl
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
names=['l'+str(i) for i in range(24)]+['head'+str(i) for i in range(8)]
for _ in range(450):
 if all((bld/('w8scaled-all' if n.startswith('l') and int(n[1:])%4!=3 else 'w8cal-all')/n/'conversion.json').exists() and json.loads((bld/('w8scaled-all' if n.startswith('l') and int(n[1:])%4!=3 else 'w8cal-all')/n/'conversion.json').read_text())['exit']==0 for n in names):break
 time.sleep(2)
else:raise RuntimeError('full compile incomplete')
stage=bld/'deploy-w8scaled';stage.mkdir(exist_ok=True);records=[]
for n in names:
 p=bld/('w8scaled-all' if n.startswith('l') and int(n[1:])%4!=3 else 'w8cal-all')/n/'export_nbg_unify/network_binary.nb';dest=stage/(n+'.nb')
 if not dest.exists():dest.symlink_to(p)
 records.append({'name':dest.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(stage/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in records));(r/'results/deployment-w8scaled-manifest.json').write_text(json.dumps(records,indent=2));print('STAGING',sum(x['bytes'] for x in records),flush=True)
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d='/opt/opi-vlm/minicpm-v46-npu-20261023';target=d+'/w8scaled-all'
def run(cmd,timeout=40):
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);print(z.stdout,z.stderr,flush=True);assert z.returncode==0,z.returncode;return z
run('mkdir -p '+target)
ssh='ssh -T '+shlex.join(b.COMMON)
cmd=['sshpass','-e','rsync','-azL','--partial','--link-dest='+d+'/w8cal-all','--no-whole-file','--checksum','--info=stats2','--rsync-path=flock -w 10 /run/opi-npu-kv-benchmark.lock rsync','-e',ssh,str(stage)+'/',b.HOST+':'+target+'/']
z=subprocess.run(cmd,env=b.env(),timeout=1200);assert z.returncode==0;print('TRANSFER_COMPLETE',flush=True)
run('cd '+target+' && sha256sum -c SHA256SUMS',120)
run('ln '+d+'/baseline-fp16/embeddings.f16 '+target+'/embeddings.f16 && ln '+d+'/baseline-fp16/prompt.i32 '+target+'/prompt.i32')
for n in ['mini_runtime_w8.c','resident_worker.py','benchmark_intern.py']:b.scp(r/'scripts'/n,b.HOST+':'+d+'/'+n,check=True,timeout=30)
b.scp(r/'results/intern-comparison-input.json',b.HOST+':'+d+'/intern-comparison-input.json',check=True,timeout=30)
run('cd '+d+' && cc -O2 -Wall -Wextra -Wno-misleading-indentation mini_runtime_w8.c -lNBGlinker -lVIPhal -lm -o mini_runtime_w8')
cmd='cd '+d+' && timeout --kill-after=10s 180s taskset -c 6,7 ./mini_runtime_w8 '+target+' '+target+'/prompt.i32 '+d+'/w8scaled-runtime.jsonl 3'
z=run(cmd,195);(r/'results/w8scaled-runtime.log').write_text(z.stdout+z.stderr)
b.scp(b.HOST+':'+d+'/w8scaled-runtime.jsonl',r/'results/w8scaled-runtime.jsonl',check=True,timeout=30)
run('df -h /; free -m');print('COMPLETE',flush=True)
