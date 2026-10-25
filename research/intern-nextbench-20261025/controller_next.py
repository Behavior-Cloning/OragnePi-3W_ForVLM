import argparse,json,os,re,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;remote='/opt/opi-vlm/intern-nextbench-20261025'
a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--config',default='config.profile.json');a.add_argument('--repeats',type=int,default=1);a.add_argument('--phases',default='performance,quality,diagnostic');a.add_argument('--timeout',type=int,default=1200);args=a.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.run)
status_file=p/(args.run+'-controller.json');assert not status_file.exists(),'run already claimed locally'
s={'run':args.run,'status':'starting','started':time.time(),'phases':args.phases,'repeats':args.repeats}
def save():
 q=status_file.with_suffix('.tmp');q.write_text(json.dumps(s,indent=2));q.replace(status_file)
def ssh(cmd,**kw):return b.ssh(cmd,timeout=30,**kw)
save();runroot=remote+'/runs/'+args.run;output=runroot+'/result';exitfile=runroot+'/exit';log=runroot+'/process.log'
try:
 with b.hostlock():
  ssh('mkdir -p '+remote+'/runs',check=True);ssh('mkdir '+runroot,check=True)
  cfg=json.loads((p/args.config).read_text())
  sources={n:p/n for n in ['board_benchmark.py','resident_worker.py','context_rag.py','aligned-inputs.json']}
  sources['resident-mask']=p.parent/'async-agent-20261025/resident-mask'
  sources['resident-profile']=p.parent/'intern-nextopt-20261025/resident-profile'
  sources['atoms.json']=p.parent/'home-companion-atoms128-v2-20261022/atoms.json'
  needed={Path(v['binary']).name for v in cfg['variants']}
  for name,rel in [('resident-tail8','runtime-tail8/resident-tail8'),('resident-greedy','runtime-greedy/resident-greedy'),('resident-combined','runtime-combined/resident-combined')]:
   if name in needed:sources[name]=p.parent/'intern-nextopt-20261025'/rel
  for n,src in sources.items():b.scp(str(src),b.HOST+':'+runroot+'/'+n,check=True,timeout=40)
  for v in cfg['variants']:
   v['binary']=runroot+'/'+Path(v['binary']).name
  cfg['rag_catalog']=runroot+'/atoms.json';cfg['fixture_path']=runroot+'/aligned-inputs.json'
  frozen=p/(args.run+'-config.json');frozen.write_text(json.dumps(cfg,ensure_ascii=False,indent=2));b.scp(str(frozen),b.HOST+':'+runroot+'/config.json',check=True,timeout=30)
  ssh('chmod 755 '+' '.join(shlex.quote(runroot+'/'+name) for name in sorted(needed|{'resident-mask','resident-profile'})),check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 '+str(args.timeout)+' /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+runroot+'/board_benchmark.py --config '+runroot+'/config.json --out '+output+' --repeats '+str(args.repeats)+' --phases '+shlex.quote(args.phases)
  shell=cmd+' >'+log+' 2>&1; rc=$?; printf "%s" "$rc" >'+exitfile
  ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True)
  s['status']='running';save();deadline=time.monotonic()+args.timeout+120
  while True:
   time.sleep(15);z=ssh('cat '+exitfile,capture_output=True,text=True)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('board timeout/exit file absent')
  ssh('mkdir -p '+output+' && cp '+log+' '+output+'/process.log',check=True)
  b.ssh('tar -czf '+runroot+'/result.tar.gz -C '+output+' .',check=True,timeout=80)
  arc=p/(args.run+'.tar.gz');b.scp(b.HOST+':'+runroot+'/result.tar.gz',str(arc),check=True,timeout=120)
  dest=p/'results'/args.run;dest.mkdir(parents=True,exist_ok=False)
  with tarfile.open(arc) as tar:
   for member in tar.getmembers():
    q=(dest/member.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents
    assert not member.issym() and not member.islnk()
   tar.extractall(dest)
  meta=dest/'meta.json';s['meta_status']=json.loads(meta.read_text())['status'] if meta.exists() else 'missing'
  s['status']='finished' if s.get('returncode')==0 and s['meta_status']=='complete' else 'failed';s['ended']=time.time();save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
