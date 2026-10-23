"""Transfer and execute one isolated full Llama NPU candidate under both locks."""
from pathlib import Path
import argparse,json,time,subprocess,shlex,hashlib,sys,os
from model import ROOT,BUILD,SOURCE_TAG,EVIDENCE,sha
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
p=argparse.ArgumentParser();p.add_argument('--stage-only',action='store_true');p.add_argument('--precision',default='attn-fp16-mlp-w8');p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run; requires complete20 graph manifest and FP32 reference.');raise SystemExit()
graphs=BUILD/a.precision;ev=EVIDENCE/a.precision
if not ev.exists() and SOURCE_TAG=='q4-source':ev=ROOT/'build-evidence'/a.precision
manifest=json.loads((graphs/'manifest.json').read_text()) if (graphs/'manifest.json').exists() else dict(graphs=[json.loads(x.read_text()) for x in sorted(ev.glob('layer[0-9]*.json'),key=lambda p:int(p.stem[5:]))]);assert a.stage_only or len(manifest['graphs'])==20
assets=BUILD/'deployment-assets';assets.mkdir(exist_ok=True)
reference=BUILD/'oracle/reference.json';assert reference.exists()
board='/opt/opi-vlm/llama32-full-npu-20261023/'+SOURCE_TAG+'/'+a.precision
stamp=time.strftime('%Y%m%dT%H%M%S');result=ROOT/('board-results-'+SOURCE_TAG+'-'+stamp);result.mkdir()
log=(result/'controller.log').open('w',buffering=1);events=(result/'transfer.jsonl').open('w',buffering=1)
lock='/run/opi-npu-kv-benchmark.lock';q=shlex.quote
files=[]
for i,row in enumerate(manifest['graphs']):
 f=graphs/f'layer{i}.nb';assert sha(f)==row['nbg_sha256'];files.append((f,'graphs/'+f.name,row['nbg_sha256']))
for name in ['embeddings.f32','rope_inv.f32','inputs.json']:
 f=BUILD/name;files.append((f,'assets/'+name,sha(f)))
files.append((reference,'assets/reference.json',sha(reference)))
for name in ['resident_session.c','full_model.c','board_probe.c','batch_prefill.h','greedy_i16.h','llama_rope.h','llama_eos.h','board_run.py','resident_worker.py']:
 f=ROOT/name;files.append((f,name,sha(f)))
if (graphs/'manifest.json').exists():files.append((graphs/'manifest.json','graphs/manifest.json',sha(graphs/'manifest.json')))
def remote(command,timeout=30):
 r=b.ssh('flock -n -E 73 '+q(lock)+' sh -c '+q(command),capture_output=True,text=True,timeout=timeout)
 log.write(r.stdout+r.stderr);log.flush()
 if r.returncode:raise RuntimeError('remote returncode '+str(r.returncode))
 return r.stdout
def transfer(src,dst,retrieve=False):
 start=time.monotonic();shell=' '.join(q(x) for x in ['sshpass','-e','ssh','-T',*b.COMMON]);args=['rsync','--checksum','--partial','--append-verify','-z','-r','--rsync-path=flock -n -E 73 '+q(lock)+' rsync','-e',shell,str(src),str(dst)]
 r=subprocess.run(args,env=b.env(),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 events.write(json.dumps(dict(source=str(src),destination=str(dst),seconds=time.monotonic()-start,bytes=Path(src).stat().st_size if not retrieve else None,returncode=r.returncode,retrieve=retrieve))+'\n')
 if r.returncode:raise RuntimeError('rsync returncode '+str(r.returncode))

status=dict(source_tag=SOURCE_TAG,board=board,result=str(result),state='starting',started=time.time())
def status_write():
 (result/'status.json').write_text(json.dumps(status,indent=2));(ROOT/('latest-controller-'+SOURCE_TAG+'.json')).write_text(json.dumps(status,indent=2))
status_write()
with b.hostlock():
 try:
  # Acquiring board flock precedes even project-directory creation.
  check=json.loads(remote("python3 -c "+q("import shutil,json;print(json.dumps(dict(free=shutil.disk_usage('/opt').free)))")))
  total=sum(f.stat().st_size for f,_,_ in files)
  existing=int(remote('du -sb '+q(board)+' 2>/dev/null | cut -f1 || true').strip() or '0')
  needed=max(0,total-existing)
  if check['free']<needed+1024**3:raise RuntimeError('Insufficient board free space for candidate plus1GiB reserve')
  remote('mkdir -p '+q(board+'/graphs')+' '+q(board+'/assets'))
  status.update(state='transferring',total_bytes=total,board_free_before=check['free']);status_write()
  for src,rel,digest in files:
   destination=board+'/'+rel;transfer(src,b.HOST+':'+destination)
   actual=remote('sha256sum '+q(destination),120).split()[0]
   if actual!=digest:raise RuntimeError('Transferred SHA mismatch '+rel)
   print('verified',rel,src.stat().st_size,flush=True)
  if a.stage_only:
   status.update(state='staged-only',graphs_staged=len(manifest['graphs']));status_write();raise SystemExit(0)
  status.update(state='compiling-runtime');status_write()
  defs=['-DMODEL_DIM=2048','-DMODEL_KV=8','-DMODEL_HD=64','-DROPE_BASE=500000.0','-DFUSED_LAYERS=1','-DKEY_CACHE_TRANSPOSED=0','-DHEAD_SHARDS=4','-DHEAD_CHUNK=32064']
  remote('cd '+q(board)+' && '+' '.join(q(x) for x in ['gcc','-O2','-Wall','-Wextra','-Werror',*defs,'resident_session.c','-lNBGlinker','-lVIPhal','-lm','-o','resident']),120)
  remote('python3 -m py_compile '+q(board+'/board_run.py')+' '+q(board+'/resident_worker.py'))
  remote('sha256sum '+q(board+'/resident'))
  status.update(state='running');status_write();out=board+'/output-'+stamp
  command=' '.join(q(x) for x in ['timeout','--signal=TERM','--kill-after=20','1800','python3','-u',board+'/board_run.py','--graphs',board+'/graphs','--assets',board+'/assets','--output',out,'--natural-max-new','96'])
  started=time.monotonic();r=b.ssh('flock -n -E 73 '+q(lock)+' '+command,stdout=log,stderr=subprocess.STDOUT,timeout=1830)
  status.update(board_returncode=r.returncode,board_seconds=time.monotonic()-started,state='retrieving');status_write()
  transfer(b.HOST+':'+out+'/',str(result)+'/',True)
  final=json.loads(remote("python3 -c "+q("import shutil,json;print(json.dumps(dict(free=shutil.disk_usage('/opt').free)))")))
  status.update(state='returned',board_free_after=final['free']);status_write()
  if r.returncode:raise RuntimeError('Board experiment returncode '+str(r.returncode))
 except Exception as e:
  status.update(state='error',error=repr(e));status_write();raise
 finally:log.close();events.close()
