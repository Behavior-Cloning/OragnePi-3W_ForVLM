"""Sole board owner: execute prepared Q8 diagnostics after full-model worker exits."""
from pathlib import Path
import json,subprocess,shlex,sys,time,hashlib,argparse
R=Path(__file__).resolve().parent;P=R.parent/'llama32-q8-quant-probe-20261023';B=Path('/opt/opi-lab/llama32-q8-quant-probe-20261023')
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run; six compiled probes and released full worker required.');raise SystemExit()
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
q=shlex.quote;lock='/run/opi-npu-kv-benchmark.lock';stamp=time.strftime('%Y%m%dT%H%M%S');board='/opt/opi-vlm/llama32-q8-quant-probe-20261023';out=P/('board-results-'+stamp);out.mkdir();log=(out/'controller.log').open('w',buffering=1)
names=['layer0-allw8','layer0-mixed']+['head'+str(i)+'-w8' for i in range(4)]
for name in names:assert (P/'evidence'/(name+'.json')).exists() and (B/'nbg'/(name+'.nb')).exists()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as d:
  for x in iter(lambda:d.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def remote(cmd,timeout=60):
 p=b.ssh('flock -n -E 73 '+q(lock)+' sh -c '+q(cmd),capture_output=True,text=True,timeout=timeout);log.write(p.stdout+p.stderr);log.flush()
 if p.returncode:raise RuntimeError('remote returncode='+str(p.returncode))
 return p.stdout
def rsync(src,dst):
 args=['rsync','-rzc','--partial','--append-verify','--rsync-path=flock -n -E 73 '+q(lock)+' rsync','-e',' '.join(q(x) for x in ['sshpass','-e','ssh','-T',*b.COMMON]),str(src),str(dst)];t=time.monotonic();p=subprocess.run(args,env=b.env(),stdout=log,stderr=subprocess.STDOUT,timeout=1200);log.write(json.dumps(dict(transfer=str(src),seconds=time.monotonic()-t,returncode=p.returncode))+'\n');log.flush();assert p.returncode==0
status=dict(state='starting',result=str(out),board=board)
def state(**v):status.update(v);(R/'q8-probe-controller-status.json').write_text(json.dumps(status,indent=2));(out/'status.json').write_text(json.dumps(status,indent=2))
state()
with b.hostlock():
 check="from pathlib import Path;import re,shutil;v=re.findall(r'\\|\\s*(0x[0-9a-fA-F]+)\\s*\\|\\s*0x[0-9a-fA-F]+\\s*\\|',Path('/sys/kernel/debug/viplite/mem_profile').read_text());assert v and int(v[-1],16)==0,v;assert shutil.disk_usage('/opt').free>1500*1024**2"
 remote('python3 -c '+q(check));remote('mkdir -p '+q(board+'/nbg')+' '+q(board+'/fixtures')+' '+q(board+'/results'))
 state(state='transferring')
 for name in names:
  src=B/'nbg'/(name+'.nb');meta=json.loads((P/'evidence'/(name+'.json')).read_text());assert sha(src)==meta['sha256'];rsync(src,b.HOST+':'+board+'/nbg/'+src.name);assert remote('sha256sum '+q(board+'/nbg/'+src.name)).split()[0]==meta['sha256']
 for name in ['quant_probe.c','board_probe.c']:
  rsync(P/name,b.HOST+':'+board+'/'+name);assert remote('sha256sum '+q(board+'/'+name)).split()[0]==sha(P/name)
 fixture=B/'fixtures';checksum=''.join(sha(f)+'  '+f.name+'\n' for f in sorted(fixture.glob('*.f32')));manifest=out/'fixtures-SHA256SUMS';manifest.write_text(checksum);rsync(str(fixture)+'/',b.HOST+':'+board+'/fixtures/');rsync(manifest,b.HOST+':'+board+'/fixtures/SHA256SUMS');remote('cd '+q(board+'/fixtures')+' && sha256sum -c SHA256SUMS',180)
 remote('cd '+q(board)+' && gcc -O2 -Wall -Wextra -Werror -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 quant_probe.c -lNBGlinker -lVIPhal -lm -o quant_probe',120)

 for name in names:
  state(state='running',variant=name)
  health="from pathlib import Path;import json;print(json.dumps(dict(temperature=max(int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')),available_kB=int([x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1]))))"
  h=json.loads(remote('python3 -c '+q(health)));assert h['temperature']<75 and h['available_kB']>256*1024,h
  dst=board+'/results/'+name+'-'+stamp;remote('mkdir -p '+q(dst));shard=-1 if name.startswith('layer') else int(name[4]);cmd=' '.join(q(x) for x in ['timeout','--signal=TERM','--kill-after=10','240',board+'/quant_probe',board+'/nbg/'+name+'.nb',board+'/fixtures',dst,str(shard)])+' >'+q(dst+'/probe.log')+' 2>&1'
  t=time.monotonic();r=b.ssh('flock -n -E 73 '+q(lock)+' sh -c '+q(cmd),stdout=log,stderr=subprocess.STDOUT,timeout=260);local=out/name;local.mkdir();rsync(b.HOST+':'+dst+'/',str(local)+'/');(local/'runner-exit.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,before=h),indent=2));assert r.returncode==0
  rows=[json.loads(x) for x in (local/'probe.log').read_text().splitlines() if x.startswith('{')];assert any(x.get('event')=='closed' and x['driver_released']==0 for x in rows);print('RETURNED',name,str(local),flush=True)
 # Matched Q4 source control uses its OWN frozen fixtures and source-specific reference.
 state(state='transferring-q4-control');qb=B/'q4-control';qboard=board+'/q4-control';remote('mkdir -p '+q(qboard+'/fixtures'))
 src=qb/'nbg/layer0-allw8.nb';digest='d43ba5cd9ba14630981e41c7bab6f51f6ab927c0f8e560f36eb898ddf7d7edfc';assert sha(src)==digest;rsync(src,b.HOST+':'+qboard+'/layer0-allw8.nb');assert remote('sha256sum '+q(qboard+'/layer0-allw8.nb')).split()[0]==digest
 fixture=qb/'fixtures';manifest=out/'q4-fixtures-SHA256SUMS';manifest.write_text(''.join(sha(f)+'  '+f.name+'\n' for f in sorted(fixture.glob('*.f32'))));rsync(str(fixture)+'/',b.HOST+':'+qboard+'/fixtures/');rsync(manifest,b.HOST+':'+qboard+'/fixtures/SHA256SUMS');remote('cd '+q(qboard+'/fixtures')+' && sha256sum -c SHA256SUMS',180)
 state(state='running-q4-control');h=json.loads(remote('python3 -c '+q(health)));assert h['temperature']<75 and h['available_kB']>256*1024,h
 dst=qboard+'/result-'+stamp;remote('mkdir -p '+q(dst));cmd=' '.join(q(x) for x in ['timeout','--signal=TERM','--kill-after=10','240',board+'/quant_probe',qboard+'/layer0-allw8.nb',qboard+'/fixtures',dst,'-1'])+' >'+q(dst+'/probe.log')+' 2>&1'
 t=time.monotonic();r=b.ssh('flock -n -E 73 '+q(lock)+' sh -c '+q(cmd),stdout=log,stderr=subprocess.STDOUT,timeout=260);local=out/'q4-control/layer0-allw8';local.mkdir(parents=True);rsync(b.HOST+':'+dst+'/',str(local)+'/');(local/'runner-exit.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,before=h),indent=2));assert r.returncode==0
 rows=[json.loads(x) for x in (local/'probe.log').read_text().splitlines() if x.startswith('{')];assert any(x.get('event')=='closed' and x['driver_released']==0 for x in rows);print('RETURNED q4-control',str(local),flush=True)
 state(state='finished');log.close()

