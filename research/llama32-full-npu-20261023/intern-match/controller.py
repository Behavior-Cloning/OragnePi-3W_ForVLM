from pathlib import Path
import sys,json,time,subprocess,shlex,hashlib,argparse
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run: same3 semantic prompts, preserved Intern graphs; explicit execute required');raise SystemExit()
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
q=shlex.quote;lock='/run/opi-npu-kv-benchmark.lock';board='/opt/opi-vlm/llama32-full-npu-20261023/intern-match';graphs='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018';stamp=time.strftime('%Y%m%dT%H%M%S');result=R/('results-'+stamp);result.mkdir();log=(result/'controller.log').open('w',buffering=1)
def status(**x):(R/'status.json').write_text(json.dumps(dict(result=str(result),**x),indent=2))
def remote(cmd,timeout=60):
 p=b.ssh('flock -n -E 73 '+q(lock)+' sh -c '+q(cmd),capture_output=True,text=True,timeout=timeout);log.write(p.stdout+p.stderr);log.flush();assert p.returncode==0;return p.stdout
def rsync(src,dst):
 p=subprocess.run(['rsync','-rzc','--rsync-path=flock -n -E 73 '+q(lock)+' rsync','-e',' '.join(q(x) for x in ['sshpass','-e','ssh','-T',*b.COMMON]),str(src),str(dst)],env=b.env(),stdout=log,stderr=subprocess.STDOUT,timeout=180);assert p.returncode==0
with b.hostlock():
 status(state='deploying');remote('mkdir -p '+q(board+'/assets'))
 files=[*R.glob('*.c'),*R.glob('*.h'),R/'board_run.py',R/'resident_worker.py',R/'inputs.json']
 for src in files:
  dst=board+('/assets/' if src.name=='inputs.json' else '/')+src.name;rsync(src,b.HOST+':'+dst);assert remote('sha256sum '+q(dst)).split()[0]==hashlib.sha256(src.read_bytes()).hexdigest()
 provenance=remote('sha256sum '+q(graphs)+'/*.nb',120);(result/'graph-sha256.txt').write_text(provenance)
 cmd='cd '+q(board)+' && gcc -O2 -Wall -Wextra -Werror -DMODEL_DIM=896 -DMODEL_KV=2 -DMODEL_HD=64 -DROPE_BASE=1000000.0 -DFUSED_LAYERS=2 -DKEY_CACHE_TRANSPOSED=0 -DHEAD_SHARDS=4 -DHEAD_CHUNK=49152 resident_session.c -lNBGlinker -lVIPhal -lm -o resident';remote(cmd,120);remote('sha256sum '+q(board+'/resident'))
 status(state='running');out=board+'/output-'+stamp;cmd=' '.join(q(x) for x in ['timeout','--signal=TERM','--kill-after=20','900','python3','-u',board+'/board_run.py','--graphs',graphs,'--assets',board+'/assets','--output',out,'--natural-max-new','96']);r=b.ssh('flock -n -E 73 '+q(lock)+' '+cmd,stdout=log,stderr=subprocess.STDOUT,timeout=930);rsync(b.HOST+':'+out+'/',str(result)+'/');status(state='returned',returncode=r.returncode);assert r.returncode==0
log.close()
