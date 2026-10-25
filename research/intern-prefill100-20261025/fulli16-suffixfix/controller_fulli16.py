#!/usr/bin/env python3
"""Stage one new graph and run C1280 A-B-A. Dry preflight unless --execute."""
from pathlib import Path
import argparse,json,re,shlex,sys,tarfile,time,traceback
from contract import sha,freeze,validate_aba,BINARY_SHA
P=Path(__file__).resolve().parent;RESEARCH=P.parents[1]
def preflight(config,run):
 assert re.fullmatch(r'[a-zA-Z0-9_-]+',run)
 raw=json.loads(config.read_text());remote='/opt/opi-vlm/intern-prefill100-20261025/fulli16-suffixfix/runs/'+run
 cfg=freeze(raw,remote)
 binary=P.parent/'suffixgroup-firstfix/runtime/resident-suffixgroup'
 assert sha(binary)==BINARY_SHA
 assert sha(P/'inputs.json')==cfg['expected_fixture_sha256']
 sources={n:P/n for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py','inputs.json']}
 sources.update({'resident-combined-v2':binary,'atoms.json':RESEARCH/'home-companion-atoms128-v2-20261022/atoms.json','prefill-layer2-i16.nb':Path(cfg['candidate']['candidate_nb'])})
 assert all(f.is_file() for f in sources.values())
 cfg['controller_source_sha256']=sha(Path(__file__))
 return cfg,remote,sources

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--config',default='config.fulli16.json');ap.add_argument('--execute',action='store_true');ap.add_argument('--timeout',type=int,default=3600);args=ap.parse_args()
 assert 120<=args.timeout<=3600
 cfg,remote,sources=preflight(P/args.config,args.run)
 if not args.execute:
  print(json.dumps({'status':'host_preflight_only_no_board_access','remote':remote,'variants':[v['name'] for v in cfg['variants']],'counts':{'performance':72,'natural':144,'diagnostic_requests':24,'diagnostic_steps':96},'candidate_sha256':cfg['candidate']['candidate_sha256'],'upload_bytes_without_config':sum(p.stat().st_size for p in sources.values())},indent=2));return
 sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
 status=P/(args.run+'-controller.json');assert not status.exists(),'unique run name required'
 output=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
 state={'run':args.run,'status':'starting','started':time.time(),'remote':remote,'timeout':args.timeout,'phases':'performance,quality,diagnostic','repeats':1}
 def save():
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status)
 def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',30),**kw)
 def upload(src,dst):
  locked('cat > '+shlex.quote(dst+'.part')+' && mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst),input=Path(src).read_bytes(),timeout=180,check=True)
 def archive_result(destname,extract):
  locked('mkdir -p '+shlex.quote(output)+' && test -f '+shlex.quote(logfile)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
  packet=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=180,capture_output=True,check=True)
  arc=P/destname;arc.write_bytes(packet.stdout)
  if not extract:return arc,None
  dest=P/'results'/args.run;dest.mkdir(parents=True,exist_ok=False)
  with tarfile.open(arc) as tar:
   for m in tar.getmembers():
    q=(dest/m.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents
    assert not m.issym() and not m.islnk()
   tar.extractall(dest)
  return arc,dest
 save()
 try:
  validate_aba(cfg)
  frozen=P/(args.run+'-config.json');frozen.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');sources['config.json']=frozen
  with b.hostlock():
   z=locked('df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True)
   free=int(z.stdout.splitlines()[-1]);need=sum(f.stat().st_size for f in sources.values())
   # Reserve768MiB after transfer plus150MiB for telemetry/24diagnosticf32.
   assert free>need+768*1024**2+150*1024**2,(free,need)
   locked('mkdir -p '+shlex.quote(str(Path(remote).parent))+' && mkdir '+shlex.quote(remote),check=True)
   state.update(status='uploading',remote_claimed=True,upload_bytes=need,source_sha256={n:sha(f) for n,f in sources.items()});save()
   for n,f in sources.items():
    started=time.monotonic();upload(f,remote+'/'+n)
    z=locked('sha256sum '+shlex.quote(remote+'/'+n),capture_output=True,text=True,check=True)
    assert z.stdout.split()[0]==sha(f),n
    print(json.dumps({'event':'uploaded','file':n,'bytes':f.stat().st_size,'seconds':time.monotonic()-started}),flush=True)
   locked('chmod 755 '+shlex.quote(remote+'/resident-combined-v2'),check=True)
   oc=cfg['overlay_contract'];locked('mkdir '+shlex.quote(oc['overlay']),check=True)
   for n,target in oc['symlinks'].items():locked('ln -s '+shlex.quote(target)+' '+shlex.quote(oc['overlay']+'/'+n),check=True)
   # Check all29 links and candidate hash under the same board lock before loading.
   verify="import json,sys;sys.path.insert(0,"+repr(remote)+");from contract import verify_overlay;print(json.dumps(verify_overlay(json.load(open("+repr(remote+'/config.json')+")))))"
   z=locked('/opt/opi-vlm/jev-lab-20261020/venv/bin/python -c '+shlex.quote(verify),capture_output=True,text=True,check=True)
   state['overlay_verified_before_launch']=json.loads(z.stdout);save()
   cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 '+str(args.timeout)+' /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+shlex.quote(remote+'/board_fulli16.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)+' --repeats 1 --phases performance,quality,diagnostic'
   shell=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
   b.ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
   state.update(status='running');save();deadline=time.monotonic()+args.timeout+180
   while True:
    time.sleep(15);z=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
    if z.returncode==0:state['returncode']=int(z.stdout.strip());break
    if time.monotonic()>deadline:raise TimeoutError('remote deadline; no automatic restart')
   arc,dest=archive_result(args.run+'.tar.gz',True)
   meta=dest/'meta.json';state['meta']=json.loads(meta.read_text()) if meta.exists() else None
   state.update(status='complete' if state.get('returncode')==0 and state['meta'] and state['meta']['status']=='complete' else 'failed',ended=time.time(),archive=str(arc));save()
 except BaseException:
  state.update(status='failed',error=traceback.format_exc(),ended=time.time(),recovery_pending=bool(state.get('remote_claimed')));save()
  if state.get('remote_claimed'):
   try:
    with b.hostlock():
     arc,_=archive_result(args.run+'-partial-'+str(int(time.time()))+'.tar.gz',False)
     state.update(recovery_pending=False,partial_archive=str(arc),recovery_note='Both locks reacquired; partial recovered; original error retained; no restart.')
   except BaseException:state['recovery_error']=traceback.format_exc()
   save()
  raise
if __name__=='__main__':main()
