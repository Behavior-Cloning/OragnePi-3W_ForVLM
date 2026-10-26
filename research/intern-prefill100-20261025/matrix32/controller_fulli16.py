#!/usr/bin/env python3
"""Stage original precision C512 banks and run M16/M32/M16-repeat. Dry preflight unless --execute."""
from pathlib import Path
import argparse,json,re,shlex,sys,tarfile,time,traceback
from contract import sha,freeze,validate_aba,BINARY_SHA
P=Path(__file__).resolve().parent;RESEARCH=P.parents[1]
def preflight(config,run):
 assert re.fullmatch(r'[a-zA-Z0-9_-]+',run)
 raw=json.loads(config.read_text());raw['artifacts']=json.loads((P/'artifacts.json').read_text())
 remote='/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/'+run
 cfg=freeze(raw,remote)
 binary=P.parent/'capacity512-runtime/runtime/resident-cap512'
 assert sha(binary)==BINARY_SHA and sha(P/'inputs.json')==cfg['expected_fixture_sha256']
 sources={n:P/n for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py','inputs.json']}
 sources.update({'resident-cap512':binary,'atoms.json':RESEARCH/'home-companion-atoms128-v2-20261022/atoms.json'})
 audits={name:json.loads((P.parent/name/'port-audit.json').read_text()) for name in ['capacity512-range16','capacity512-original-l2','capacity512-m32']}
 for name,a in raw['artifacts'].items():
  path=Path(a['local_path']);assert path.is_file() and path.stat().st_size==a['bytes'] and sha(path)==a['sha256']
  family='capacity512-m32' if name.startswith('m32-') else ('capacity512-original-l2' if name in ['m16-l2.nb','m8-l2.nb'] else 'capacity512-range16')
  cases=[x for x in audits[family]['cases'] if x['nb_sha256']==a['sha256']]
  assert len(cases)==1 and cases[0]['payload_unchanged'] and cases[0]['quantizer_unchanged']
  assert cases[0].get('actual_port_capacity_only') or cases[0].get('actual_port_capacity_and_batch_only')
  if name=='m32-l2.nb':assert '/original32/l2/' in a['local_path']
  sources['artifacts/'+name]=path
 assert len(raw['artifacts'])==48 and all(f.is_file() for f in sources.values())
 cfg['compile_port_audit_sha256']={n:sha(P.parent/n/'port-audit.json') for n in audits}
 cfg['controller_source_sha256']=sha(Path(__file__))
 return cfg,remote,sources

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--config',default='config.fulli16.json');ap.add_argument('--execute',action='store_true');ap.add_argument('--timeout',type=int,default=3600);args=ap.parse_args()
 assert 120<=args.timeout<=3600
 cfg,remote,sources=preflight(P/args.config,args.run)
 if not args.execute:
  print(json.dumps({'status':'host_preflight_only_no_board_access','remote':remote,'variants':[v['name'] for v in cfg['variants']],'counts':{'performance':72,'natural':144,'diagnostic_requests':24,'diagnostic_steps':96},'capacity':512,'artifact_count':48,'upload_bytes_without_config':sum(p.stat().st_size for p in sources.values())},indent=2));return
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
   # Stream tar through stdin; no duplicate board staging archive and no 1GB Python buffer.
   bundle=P/(args.run+'-upload.tar')
   with tarfile.open(bundle,'w') as tar:
    for n,f in sources.items():tar.add(f,arcname=n,recursive=False)
   state['bundle_sha256']=sha(bundle);save()
   with bundle.open('rb') as stream:
    locked('tar -xf - -C '+shlex.quote(remote),stdin=stream,timeout=1200,check=True)
   for n,f in sources.items():
    z=locked('sha256sum '+shlex.quote(remote+'/'+n),capture_output=True,text=True,check=True)
    assert z.stdout.split()[0]==state['source_sha256'][n],n
   locked('chmod 755 '+shlex.quote(remote+'/resident-cap512'),check=True)
   for bank,links in cfg['overlay_contract'].items():
    locked('mkdir '+shlex.quote(remote+'/'+bank),check=True)
    cmd=' && '.join('ln -s '+shlex.quote(t)+' '+shlex.quote(remote+'/'+bank+'/'+n) for n,t in links.items())
    locked(cmd,check=True)
   # Check all48 artifacts and70 links under board lock before loading.
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
  state.update(status='observation_failed' if state.get('status')=='running' and 'returncode' not in state else 'failed',error=traceback.format_exc(),ended=time.time(),recovery_pending=bool(state.get('remote_claimed')));save()
  if state.get('remote_claimed'):
   try:
    with b.hostlock():
     arc,_=archive_result(args.run+'-partial-'+str(int(time.time()))+'.tar.gz',False)
     state.update(recovery_pending=False,partial_archive=str(arc),recovery_note='Both locks reacquired; partial recovered; original error retained; no restart.')
   except BaseException:state['recovery_error']=traceback.format_exc()
   save()
  raise
if __name__=='__main__':main()
