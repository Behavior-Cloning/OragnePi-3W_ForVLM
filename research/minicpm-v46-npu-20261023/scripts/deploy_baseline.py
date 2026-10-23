from pathlib import Path
import sys,subprocess,json,shlex,hashlib,time
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
manifest=json.loads((r/'results/full-compile-status.json').read_text());assert len(manifest)==32 and all(x['exit']==0 for x in manifest)
stage=bld/'deploy-baseline';stage.mkdir(exist_ok=True);files={x['name']+'.nb':bld/'compiled'/x['name']/'export_nbg_unify/network_binary.nb' for x in manifest};files.update({'embeddings.f16':bld/'full/embeddings.f16','prompt.i32':bld/'full/prompt.i32'})
records=[]
for n,p in files.items():
 dest=stage/n
 if not dest.exists():dest.symlink_to(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 records.append({'name':n,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(stage/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in records));(r/'results/deployment-manifest.json').write_text(json.dumps(records,indent=2));print('STAGING',sum(x['bytes'] for x in records),flush=True)
# Wait for our immediately preceding diagnostic to release the shared board lock.
with b.hostlock():
 d='/opt/opi-vlm/minicpm-v46-npu-20261023';target=d+'/baseline-fp16'
 def run(cmd,timeout=40):
  z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);print(z.stdout,z.stderr,flush=True);assert z.returncode==0,z.returncode;return z
 run('mkdir -p '+target)
 ssh='ssh -T '+shlex.join(b.COMMON)
 cmd=['sshpass','-e','rsync','-azL','--partial','--append-verify','--info=stats2','--rsync-path=flock -w 10 /run/opi-npu-kv-benchmark.lock rsync','-e',ssh,str(stage)+'/',b.HOST+':'+target+'/']
 z=subprocess.run(cmd,env=b.env(),timeout=1800);assert z.returncode==0;print('TRANSFER_COMPLETE',flush=True)
 run('cd '+target+' && sha256sum -c SHA256SUMS',120)
 for n in ['mini_runtime.c']:b.scp(r/'scripts'/n,b.HOST+':'+d+'/'+n,check=True,timeout=30)
 run('cd '+d+' && cc -O2 -Wall -Wextra -Wno-misleading-indentation mini_runtime.c -lNBGlinker -lVIPhal -lm -o mini_runtime')
 cmd='cd '+d+' && timeout --kill-after=10s 240s taskset -c 6,7 ./mini_runtime '+target+' '+target+'/prompt.i32 '+d+'/baseline-runtime.jsonl 3'
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=255);(r/'results/baseline-runtime.log').write_text(z.stdout+z.stderr);print('RUNTIME_EXIT',z.returncode,flush=True)
 b.scp(b.HOST+':'+d+'/baseline-runtime.jsonl',r/'results/baseline-runtime.jsonl',check=True,timeout=30)
 run('df -h /; free -m')
