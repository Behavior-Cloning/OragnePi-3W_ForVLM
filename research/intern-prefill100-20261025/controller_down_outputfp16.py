#!/usr/bin/env python3
"""Root-executed output-FP16 probe; original / prior I16 / output-FP16 / original.
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
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('candidate_status');ap.add_argument('--execute',action='store_true',help='Without this flag only prepare locally, no board operations');a=ap.parse_args();assert a.run.replace('-','').replace('_','').isalnum()
 status=R/(a.run+'-controller.json');frozen=R/(a.run+'-config.json');assert not status.exists(),'unique run name required'
 remote='/opt/opi-vlm/intern-prefill100-20261025/runs/'+a.run;output=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
 s={'run':a.run,'status':'starting','started':time.time(),'scope':'single fused2 output-FP16 L2/M32 API original/prior-I16/new-output-FP16/original, not full-model TTFT, no audio/RAG residency','remote':remote}
 def save():
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(s,indent=2)+'\n');temp.replace(status)
 try:
  old=json.loads((R/'single-block-c512-r1-config.json').read_text());base=next(x for x in old['sequence'] if x['run_id']=='c512-m32-l2')
  assert old['binary_sha256']==PIN and sha(R/'runtime/tensor-probe-prefill100')==PIN
  assert base['capacity']==512 and base['query_tokens']==32 and base['first_layer']==2
  prior='/opt/opi-vlm/intern-prefill100-20261025/runs/single-block-c512-r1/'
  assert base['nb'].startswith(prior) and base['input'].startswith(prior) and old['binary'].startswith(prior)
  csfile=Path(a.candidate_status).resolve();assert R in csfile.parents
  cs=json.loads(csfile.read_text());assert cs['state']=='compiled_for_single_block_screen_not_quality_qualified' and cs['actual_ports_exact']
  geo=cs.get('geometry',cs)
  assert geo['first_layer']==2 and geo['fusion_width']==2 and geo['capacity']==512 and geo['query_tokens']==32
  auditfile=csfile.parent/'static-delivery-audit.json';assert sha(auditfile)==cs['static_audit_sha256']
  audit=json.loads(auditfile.read_text());assert audit['actual_ports_exact'] and audit['unchanged_weight_count']==12 and audit['new_i16_weights']==2
  assert len(audit['weights'])==14
  assert all(x['same_bytes'] if 'fullconnect_116' not in x['key'] else x['candidate_type']=='INT16' and (x['quantized_source_match'] or x['transpose_match']) for x in audit['weights'])
  assert cs.get('output_policy') and audit.get('output_fp16')
  independent=csfile.parent/'INDEPENDENT_STATIC_AUDIT.json';assert sha(independent)==cs['independent_static_audit_sha256']==audit['independent_static_audit_sha256']
  independent_data=json.loads(independent.read_text());assert independent_data['status']=='STATIC_CHECKS_PASS_NOT_BOARD_QUALIFIED'
  assert cs['nb_sha256']==independent_data['candidate_nb_sha256'] and cs['nb_bytes']==independent_data['candidate_nb_bytes']
  previous_status=R/'graphs/candidates-down-i16-range16/deployment-manifest.json';previous=json.loads(previous_status.read_text());previous_nb=Path(previous['nb_path']);assert sha(previous_nb)==previous['nb_sha256']
  assert previous['geometry']=={'capacity':512,'query_tokens':32,'first_layer':2,'fusion_width':2} and previous['actual_ports_exact']
  assert previous['source_nb_sha256']==cs['source_nb_sha256']
  assert previous['nb_sha256']=='b5014320464596fb92593a962a08b9d2a14b13f9e8ee23c8579bf44c8af62cd4'
  assert sha(previous_status.parent/'static-delivery-audit.json')==previous['static_audit_sha256']
  candidate=Path(cs['nb_path']).resolve();assert R in candidate.parents and candidate.is_file()
  assert sha(candidate)==cs['nb_sha256']
  fixture=R/'fixtures-canonical/c512-m32-l2';reference=fixture/'probe-reference.f32';inp=fixture/'probe-input.f32';meta=json.loads((fixture/'probe-fixture.json').read_text())
  assert sha(inp)==base['input_sha256']==meta['input_sha256']==previous['input_sha256'] and sha(reference)==base['reference_sha256']==meta['reference_sha256']==previous['reference_sha256']
  original=R/'graphs/cases/c512-m32-l2/export_nbg_unify/network_binary.nb';assert sha(original)==base['nb_sha256']==cs['source_nb_sha256']
  shared=remote
  input_remote='/dev/shm/opi-prefill100-'+a.run+'-input.f32'
  sequence=[]
  for case,variant,nb,nbsha in [('original-before','original-fp16-down',shared+'/original.nb',base['nb_sha256']),('prior-i16','prior-down-i16-range16',remote+'/prior-i16.nb',previous['nb_sha256']),('candidate','down-i16-output-fp16',remote+'/candidate.nb',cs['nb_sha256']),('original-after','original-fp16-down-repeat',shared+'/original.nb',base['nb_sha256'])]:
   sequence.append({**base,'run_id':case,'variant':variant,'nb':nb,'nb_sha256':nbsha,'input':input_remote,'local_reference':str(reference),'reference_sha256':sha(reference)})
  runner=R/'board_probe_suite.py';sources={'candidate.nb':candidate,'prior-i16.nb':previous_nb,'original.nb':original,'input.f32':inp,'tensor-probe':R/'runtime/tensor-probe-prefill100','board_probe_suite.py':runner}
  cfg={'scope':s['scope'],'phase':'timing-only; NPD and IO profiling forbidden','binary':shared+'/tensor-probe','binary_sha256':PIN,'warmup':3,'repeats':30,'sequence':sequence,'canonical_input_sha256':sha(inp),'canonical_reference_sha256':sha(reference),'fixture_meta_sha256':sha(fixture/'probe-fixture.json'),'candidate_status_sha256':sha(csfile),'previous_status_sha256':sha(previous_status),'previous_nb_sha256':sha(previous_nb),'independent_static_audit_sha256':sha(independent),'source_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),runner]},'shared_file_verification':{},'candidate_status':str(csfile),'reserve_bytes':1024**3+100*1024**2,'input_storage':{'path':input_remote,'kind':'tmpfs','purpose':'same canonical input preloaded before all timed API calls; avoids duplicate disk use','min_mem_available_after_upload':256*1024**2},'diagnostic_policy':'Run any NPD/IO diagnostics separately in another unique directory; no diagnostic data admitted to timing stats'}
  frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen
  destinations={name:(input_remote if name=='input.f32' else remote+'/'+name) for name in sources}
  frozen_sources={name:sha(src) for name,src in sources.items()}
  def verify_frozen():
   for name,src in sources.items():assert sha(src)==frozen_sources[name],name
   for name,h in cfg['source_sha256'].items():assert sha(R/name)==h,name
   assert sha(csfile)==cfg['candidate_status_sha256'] and sha(previous_status)==cfg['previous_status_sha256'] and sha(independent)==cfg['independent_static_audit_sha256']
  verify_frozen()
  if not a.execute:
   print(json.dumps({'prepared':True,'config':str(frozen),'board_operations':0,'upload_bytes':sum(p.stat().st_size for p in sources.values())}));return
  parent=json.loads((R/'capacity512-conv-full/c512-m32-conv-aba-r1-controller.json').read_text())
  assert parent['status']=='complete' and parent['returncode']==0 and parent['meta']['status']=='complete','Current Conv full run must finish/recover; this controller does not queue a second job'
  save()
  with b.hostlock():
   verify_frozen()
   p=locked('df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True);free=int(p.stdout.splitlines()[-1]);need=sum(x.stat().st_size for name,x in sources.items() if name!='input.f32');assert free>need+cfg['reserve_bytes'],(free,need,cfg['reserve_bytes'])
   p=locked('test ! -e '+shlex.quote(input_remote)+' && test ! -e '+shlex.quote(input_remote+'.part')+' && df -B1 --output=avail /dev/shm && cat /proc/meminfo',capture_output=True,text=True,check=True)
   lines=p.stdout.splitlines();tmpfree=int(lines[1]);memfree=int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024
   assert tmpfree>2*inp.stat().st_size+16*1024**2 and memfree>inp.stat().st_size+256*1024**2,(tmpfree,memfree)
   s.update(input_tmpfs_free_before=tmpfree,mem_available_before=memfree,input_tmpfs_path=input_remote);save()
   locked('mkdir -p '+shlex.quote(str(Path(remote).parent))+' && mkdir '+shlex.quote(remote),check=True);s.update(status='uploading',remote_claimed=True,disk_upload_bytes=need,tmpfs_input_bytes=inp.stat().st_size);save()
   verify_frozen();s['upload_source_sha256']=frozen_sources;save()
   for name,src in sources.items():
    assert sha(src)==frozen_sources[name],name;upload(src,destinations[name]);assert sha(src)==frozen_sources[name],name
   verify=''.join(frozen_sources[name]+'  '+destinations[name]+'\n' for name in sources).encode();locked('sha256sum -c -',input=verify,capture_output=True,check=True)
   locked('chmod 755 '+shlex.quote(remote+'/tensor-probe'),check=True)
   cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 900 python3 -u '+shlex.quote(remote+'/board_probe_suite.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)
   script=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
   b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20);s.update(status='running');save();deadline=time.monotonic()+1020
   while True:
    time.sleep(5);p=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
    if p.returncode==0:s['returncode']=int(p.stdout.strip());break
    if time.monotonic()>deadline:raise TimeoutError('remote deadline; never auto-restart')
   locked('mkdir -p '+shlex.quote(output)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
   p=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True);archive=R/(a.run+'.tar.gz');archive.write_bytes(p.stdout);dest=R/'results'/a.run;extract(archive,dest)
   meta=dest/'meta.json';s['meta']=json.loads(meta.read_text()) if meta.exists() else None
   succeeded=s.get('returncode')==0 and s['meta'] and s['meta']['status']=='complete'
   if succeeded:
    locked('sha256sum -c - && rm -- '+shlex.quote(input_remote),input=(sha(inp)+'  '+input_remote+'\n').encode(),capture_output=True,check=True);s['input_tmpfs_removed']=True
   else:s['input_tmpfs_preserved']=True
   s.update(status='complete' if succeeded else 'failed',ended=time.time());save()
 except BaseException:
  original=traceback.format_exc();s.update(status='failed',ended=time.time(),error=original,recovery_pending=bool(s.get('remote_claimed')));save()
  if s.get('remote_claimed'):
   try:
    with b.hostlock():
     locked('test -d '+shlex.quote(remote)+' && mkdir -p '+shlex.quote(output)+' && test -f '+shlex.quote(logfile)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
     packet=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True);partial=R/(a.run+'-partial-'+str(int(time.time()))+'.tar.gz');partial.write_bytes(packet.stdout);s.update(recovery_pending=False,partial_archive=str(partial),recovery_note='Both locks acquired and partial evidence copied; original failure retained, no restart. If uploaded, tmpfs input is preserved at input_tmpfs_path for audited cleanup.')
   except BaseException:s['recovery_error']=traceback.format_exc()
   save()
  raise
if __name__=='__main__':main()
