#!/usr/bin/env python3
"""C1280 baseline controller. Run only by the board owner; two locks, unique run, partial recovery."""
from pathlib import Path
import argparse,hashlib,json,re,shlex,sys,tarfile,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
P=Path(__file__).resolve().parent
RESEARCH=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--config',default='config.fullbaseline.json');a.add_argument('--repeats',type=int,default=2);a.add_argument('--phases',default='performance,quality');a.add_argument('--timeout',type=int,default=1800);args=a.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.run) and args.repeats>=1 and 120<=args.timeout<=7200
assert set(args.phases.split(',')) <= {'performance','quality','diagnostic'}
status=P/(args.run+'-controller.json');assert not status.exists(),'unique run name required'
remote='/opt/opi-vlm/intern-prefill100-20261025/fullbaseline/runs/'+args.run
output=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
s={'run':args.run,'status':'starting','started':time.time(),'repeats':args.repeats,'phases':args.phases,'remote':remote}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save():
 q=status.with_suffix('.tmp');q.write_text(json.dumps(s,indent=2)+'\n');q.replace(status)
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',30),**kw)
def upload(src,dst):
 locked('cat > '+shlex.quote(dst+'.part')+' && mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst),input=Path(src).read_bytes(),timeout=90,check=True)
def archive_result(destname,extract):
 locked('mkdir -p '+shlex.quote(output)+' && test -f '+shlex.quote(logfile)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
 packet=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=120,capture_output=True,check=True)
 arc=P/destname;arc.write_bytes(packet.stdout)
 if extract:
  dest=P/'results'/args.run;dest.mkdir(parents=True,exist_ok=False)
  with tarfile.open(arc) as tar:
   for member in tar.getmembers():
    q=(dest/member.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents
    assert not member.issym() and not member.islnk()
   tar.extractall(dest)
  return arc,dest
 return arc,None
save()
try:
 cfg=json.loads((P/args.config).read_text());assert len(cfg['variants'])==1
 v=cfg['variants'][0];assert cfg['defaults']['capacity']==v['capacity']==1280
 assert v['graph']=='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018'
 assert len(v['expected_graph_sha256'])==29
 assert v['env']=={'OPI_TTFT_TAIL_MIN':'8','OPI_GROUP_PREFILL':'1','OPI_MASK_TEMPLATE':'1','OPI_GREEDY_I16_SINGLE_PASS':'1'}
 binary=RESEARCH/'intern-nextopt-20261025/runtime-combined-v2/resident-combined-v2'
 assert sha(binary)==v['expected_binary_sha256']=='ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d'
 assert sha(P/'inputs.json')==cfg['expected_fixture_sha256']
 sources={n:P/n for n in ['board_fullbaseline.py','resident_worker_v1.py','context_rag.py','inputs.json']}
 sources['resident-combined-v2']=binary
 sources['atoms.json']=RESEARCH/'home-companion-atoms128-v2-20261022/atoms.json'
 assert all(x.is_file() for x in sources.values())
 cfg['variants'][0]['binary']=remote+'/resident-combined-v2';cfg['fixture_path']=remote+'/inputs.json';cfg['rag_catalog']=remote+'/atoms.json'
 cfg['repeats']=args.repeats;cfg['quality']='quality' in args.phases.split(',');cfg['diagnostic']='diagnostic' in args.phases.split(',')
 cfg['controller_source_sha256']=sha(Path(__file__))
 frozen=P/(args.run+'-config.json');frozen.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');sources['config.json']=frozen
 with b.hostlock():
  z=locked('df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True)
  free=int(z.stdout.splitlines()[-1]);need=sum(x.stat().st_size for x in sources.values());assert free>need+250*1024*1024,(free,need)
  locked('mkdir -p '+shlex.quote(str(Path(remote).parent))+' && mkdir '+shlex.quote(remote),check=True)
  s.update(status='uploading',remote_claimed=True,upload_bytes=need,source_sha256={n:sha(src) for n,src in sources.items()});save()
  for name,src in sources.items():upload(src,remote+'/'+name)
  locked('chmod 755 '+shlex.quote(remote+'/resident-combined-v2'),check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 '+str(args.timeout)+' /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+shlex.quote(remote+'/board_fullbaseline.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)+' --repeats '+str(args.repeats)+' --phases '+shlex.quote(args.phases)
  shell=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
  b.ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
  s.update(status='running');save();deadline=time.monotonic()+args.timeout+180
  while True:
   time.sleep(15)
   z=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('remote deadline; do not restart automatically')
  arc,dest=archive_result(args.run+'.tar.gz',True)
  meta=dest/'meta.json';s['meta']=json.loads(meta.read_text()) if meta.exists() else None
  s.update(status='complete' if s.get('returncode')==0 and s['meta'] and s['meta']['status']=='complete' else 'failed',ended=time.time(),archive=str(arc));save()
except BaseException:
 original=traceback.format_exc();s.update(status='failed',error=original,ended=time.time(),recovery_pending=bool(s.get('remote_claimed')));save()
 if s.get('remote_claimed'):
  try:
   with b.hostlock():
    arc,_=archive_result(args.run+'-partial-'+str(int(time.time()))+'.tar.gz',False)
    s.update(recovery_pending=False,partial_archive=str(arc),recovery_note='Both locks reacquired; partial recovered; original error retained; no restart.')
  except BaseException:s['recovery_error']=traceback.format_exc()
  save()
 raise
