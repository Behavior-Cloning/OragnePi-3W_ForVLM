#!/usr/bin/env python3
"""Root-executed FP16 FCL/Conv/FCL timing only; immutable single-block A-B-A.
Preparing/importing this file never starts work; run explicitly with unique RUN.
Uses unchanged board_probe_suite.py, which clears all OPI_ profiling flags.
"""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,tarfile,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
R=Path(__file__).resolve().parent
PIN='eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',30),**kw)
def upload(src,dst):locked('cat > '+shlex.quote(dst+'.part')+' && mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst),input=Path(src).read_bytes(),timeout=180,check=True)
def extract(archive,dest):
 dest.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   q=(dest/member.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents
   assert not member.issym() and not member.islnk()
  tar.extractall(dest)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--prepare-only',action='store_true',help='freeze local config only; performs no SSH or board operations');a=ap.parse_args();assert a.run.replace('-','').replace('_','').isalnum()
 status=R/(a.run+'-controller.json');frozen=R/(a.run+'-config.json');assert not status.exists() and not frozen.exists(),'unique run name required'
 remote='/opt/opi-vlm/intern-prefill100-20261025/runs/'+a.run;output=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
 s={'run':a.run,'status':'starting','started':time.time(),'scope':'single fused2 L2/M16 API A-B-A, not full-model TTFT, no audio/RAG residency','remote':remote}
 def save():
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(s,indent=2)+'\n');temp.replace(status)
 try:
  old=json.loads((R/'single-block-c512-r1-config.json').read_text());base=next(x for x in old['sequence'] if x['run_id']=='c512-m16-l2')
  assert old['binary_sha256']==PIN and sha(R/'runtime/tensor-probe-prefill100')==PIN
  assert base['capacity']==512 and base['query_tokens']==16 and base['first_layer']==2
  prior='/opt/opi-vlm/intern-prefill100-20261025/runs/single-block-c512-r1/'
  assert base['nb'].startswith(prior) and base['input'].startswith(prior) and old['binary'].startswith(prior)
  candidate=R/'graphs/candidates-fp16/conv1x1-m16-l2/export_nbg_unify/network_binary.nb';cs=json.loads((R/'graphs/candidates-fp16/status.json').read_text())
  assert cs['state']=='complete' and cs['quantizer_byte_equal'] and cs['all_external_ports_exact']
  assert all(x['source_exported_weight_exact'] for x in cs['all_14_exported_fc_weight_dtype_and_bytes_same'])
  assert sha(candidate)==cs['nb_sha256']=='425bc11857e5691b30abf3d60700dcc735dd13f6b12bed3671f32a287dfe02b5'
  fixture=R/'fixtures-canonical/c512-m16-l2';reference=fixture/'probe-reference.f32';inp=fixture/'probe-input.f32';meta=json.loads((fixture/'probe-fixture.json').read_text())
  assert sha(inp)==base['input_sha256']==meta['input_sha256'] and sha(reference)==base['reference_sha256']==meta['reference_sha256']
  original=R/'graphs/cases/c512-m16-l2/export_nbg_unify/network_binary.nb';assert sha(original)==base['nb_sha256']
  sequence=[]
  for case,variant,nb,nbsha in [('original-before','fp16-fcl',base['nb'],base['nb_sha256']),('conv1x1','fp16-conv1x1',remote+'/candidate.nb',cs['nb_sha256']),('original-after','fp16-fcl-repeat',base['nb'],base['nb_sha256'])]:
   sequence.append({**base,'run_id':case,'variant':variant,'nb':nb,'nb_sha256':nbsha,'local_reference':str(reference),'reference_sha256':sha(reference)})
  runner=R/'board_probe_suite.py';sources={'candidate.nb':candidate,'board_probe_suite.py':runner}
  cfg={'scope':s['scope'],'phase':'timing-only; NPD and IO profiling forbidden','binary':old['binary'],'binary_sha256':PIN,'warmup':3,'repeats':30,'sequence':sequence,'canonical_input_sha256':sha(inp),'canonical_reference_sha256':sha(reference),'fixture_meta_sha256':sha(fixture/'probe-fixture.json'),'candidate_status_sha256':sha(R/'graphs/candidates-fp16/status.json'),'source_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),runner,R/'analyze_fp16conv.py']},'shared_file_verification':{old['binary']:PIN,base['nb']:base['nb_sha256'],base['input']:base['input_sha256']},'diagnostic_policy':'Run any NPD/IO diagnostics separately in another unique directory; no diagnostic data admitted to timing stats'}
  frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen
  if a.prepare_only:
   print(json.dumps({'prepared':True,'config':str(frozen),'board_operations':0,'upload_bytes':sum(p.stat().st_size for p in sources.values())}));return
  save()
  with b.hostlock():
   p=locked('df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True);free=int(p.stdout.splitlines()[-1]);need=sum(x.stat().st_size for x in sources.values());assert free>need+128*1024**2,(free,need)
   verify=''.join(h+'  '+p+'\n' for p,h in cfg['shared_file_verification'].items()).encode();locked('sha256sum -c -',input=verify,capture_output=True,check=True)
   locked('mkdir -p '+shlex.quote(str(Path(remote).parent))+' && mkdir '+shlex.quote(remote),check=True);s.update(status='uploading',remote_claimed=True,upload_bytes=need);save()
   for name,src in sources.items():upload(src,remote+'/'+name)
   verify=''.join(sha(src)+'  '+remote+'/'+name+'\n' for name,src in sources.items()).encode();locked('sha256sum -c -',input=verify,capture_output=True,check=True)
   cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 900 python3 -u '+shlex.quote(remote+'/board_probe_suite.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)
   script=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
   b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20);s.update(status='running');save();deadline=time.monotonic()+1020
   while True:
    time.sleep(5);p=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
    if p.returncode==0:s['returncode']=int(p.stdout.strip());break
    if time.monotonic()>deadline:raise TimeoutError('remote deadline; never auto-restart')
   locked('mkdir -p '+shlex.quote(output)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
   p=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True);archive=R/(a.run+'.tar.gz');archive.write_bytes(p.stdout);dest=R/'results'/a.run;extract(archive,dest)
   meta=dest/'meta.json';s['meta']=json.loads(meta.read_text()) if meta.exists() else None;s.update(status='complete' if s.get('returncode')==0 and s['meta'] and s['meta']['status']=='complete' else 'failed',ended=time.time());save()
 except BaseException:
  original=traceback.format_exc();s.update(status='failed',ended=time.time(),error=original,recovery_pending=bool(s.get('remote_claimed')));save()
  if s.get('remote_claimed'):
   try:
    with b.hostlock():
     locked('test -d '+shlex.quote(remote)+' && mkdir -p '+shlex.quote(output)+' && test -f '+shlex.quote(logfile)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
     packet=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True);partial=R/(a.run+'-partial-'+str(int(time.time()))+'.tar.gz');partial.write_bytes(packet.stdout);s.update(recovery_pending=False,partial_archive=str(partial),recovery_note='Both locks acquired and partial evidence copied; original failure retained, no restart.')
   except BaseException:s['recovery_error']=traceback.format_exc()
   save()
  raise
if __name__=='__main__':main()
