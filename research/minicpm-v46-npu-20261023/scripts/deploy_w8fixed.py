from pathlib import Path
import sys,subprocess,json,shlex,hashlib,time,fcntl
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
label='w8fixed';names=['l'+str(i) for i in range(24)]+['head'+str(i) for i in range(8)]
def source(n):return bld/('w8fixed' if n.startswith('l') and int(n[1:])%4!=3 else 'w8matint-f16' if n.startswith('l') else 'w8cal-all')/n
for _ in range(300):
 if all((source(n)/'conversion.json').exists() and json.loads((source(n)/'conversion.json').read_text())['exit']==0 for n in names):break
 time.sleep(2)
else:raise RuntimeError('compile incomplete')
stage=bld/('deploy-'+label);stage.mkdir(exist_ok=True);records=[]
for n in names:
 p=source(n)/'export_nbg_unify/network_binary.nb';dest=stage/(n+'.nb')
 assert p.is_file()
 if not dest.exists():dest.symlink_to(p)
 records.append({'name':dest.name,'source':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(stage/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in records));(r/'results'/('deployment-'+label+'-manifest.json')).write_text(json.dumps(records,indent=2))
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d='/opt/opi-vlm/minicpm-v46-npu-20261023';target=d+'/'+label+'-all'
def run(cmd,timeout=40):
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);print(z.stdout,z.stderr,flush=True);assert z.returncode==0,z.returncode;return z
run('mkdir -p '+target)
cmd=['sshpass','-e','rsync','-azL','--partial','--link-dest='+d+'/w8scaled-all','--no-whole-file','--checksum','--info=stats2','--rsync-path=flock -w 10 /run/opi-npu-kv-benchmark.lock rsync','-e','ssh -T '+shlex.join(b.COMMON),str(stage)+'/',b.HOST+':'+target+'/']
z=subprocess.run(cmd,env=b.env(),timeout=900);assert z.returncode==0
run('cd '+target+' && sha256sum -c SHA256SUMS',120)
run('ln '+d+'/baseline-fp16/embeddings.f16 '+target+'/embeddings.f16 && ln '+d+'/baseline-fp16/prompt.i32 '+target+'/prompt.i32')
for p in [r/'scripts/monitor_command.py',r/'scripts/benchmark_intern_same_text.py',r/'results/intern-same-text-input.json']:b.scp(p,b.HOST+':'+d+'/'+p.name,check=True,timeout=30)
commands=[('runtime','taskset -c 6,7 ./mini_runtime_grouped '+target+' '+target+'/prompt.i32 '+d+'/'+label+'-runtime.jsonl 3'),('holdouts','env MINI_PROMPTS_DIR='+d+'/holdout-prompts taskset -c 6,7 ./mini_runtime_cases '+target+' '+d+'/holdout-prompts/0.i32 '+d+'/'+label+'-holdouts.jsonl 4')]
for suffix,cmd in commands:
 z=run('cd '+d+' && timeout --kill-after=10s 180s python3 monitor_command.py '+label+'-'+suffix+'-telemetry.jsonl '+cmd,195)
 (r/'results'/(label+'-'+suffix+'.log')).write_text(z.stdout+z.stderr)
 for n in [label+'-'+suffix+'.jsonl',label+'-'+suffix+'-telemetry.jsonl']:b.scp(b.HOST+':'+d+'/'+n,r/'results'/n,check=True,timeout=30)
run('cd '+d+' && timeout --kill-after=10s 90s python3 benchmark_intern_same_text.py',100)
b.scp(b.HOST+':'+d+'/intern-same-text.jsonl',r/'results/intern-same-text.jsonl',check=True,timeout=30)
run('df -h /; free -m');print('COMPLETE',flush=True)
