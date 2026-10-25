#!/usr/bin/env python3
"""Root-only 48-window original/I16 numerical screen, using existing verified NBs."""
from pathlib import Path
import json,hashlib,tarfile,time,shlex,traceback,math
from controller_down_i16 import R,sha,locked,upload,extract,b,PIN
RUN='down-i16-range16-semantic-r1'
D=R/'down-i16-semantic'; remote='/opt/opi-vlm/intern-prefill100-20261025/runs/'+RUN
shared='/opt/opi-vlm/intern-prefill100-20261025/runs/down-i16-c512-m32-r1'
tmp='/dev/shm/opi-'+RUN; output=tmp+'/result'
status=R/(RUN+'-controller.json'); assert not status.exists()
s={'status':'preparing','started':time.time(),'scope':'48 debug windows paired original and range16 I16; tuned on this screen, not independent quality or TTFT','remote':remote}
def save():
 p=status.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(status)
def collect():
 locked('mkdir -p '+shlex.quote(output)+' && cp '+shlex.quote(remote+'/process.log')+' '+shlex.quote(output+'/process.log'),check=True)
 p=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=120,capture_output=True,check=True)
 archive=R/(RUN+'.tar.gz');archive.write_bytes(p.stdout);extract(archive,R/'results'/RUN)
try:
 save();st=json.loads((D/'status.json').read_text());assert st['status']=='complete' and len(st['cases'])==48
 m=json.loads((R/'graphs/candidates-down-i16-range16/deployment-manifest.json').read_text());assert m['actual_ports_exact']
 old=json.loads((R/'down-i16-c512-m32-r1-config.json').read_text())
 assert old['binary_sha256']==PIN and old['sequence'][0]['nb_sha256']==m['source_nb_sha256']
 g=json.loads((R/'graphs/cases/c512-m32-l2/decoder.json').read_text());conv=json.loads((R/'graphs/cases/c512-m32-l2/conversion.json').read_text())
 expected=4*sum(math.prod(g['Layers'][k]['parameters']['shape']) for k in conv['input_order'])
 sequence=[];checks=[];total=0
 archive=R/(RUN+'-inputs.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for f in st['cases']:
   p=D/f['id']/'input.f32';ref=D/f['id']/'reference.f32'
   assert p.stat().st_size==expected and ref.stat().st_size==180224
   assert sha(p)==f['input_sha256'] and sha(ref)==f['reference_sha256']
   assert f['history']+32<=f['prompt_tokens'] and f['history']<=512
   assert f['prompt_tokens']==f['static_tokens']+f['body_tokens']+8
   tar.add(p,arcname=f['id']+'.f32');total+=p.stat().st_size
   inp=tmp+'/'+f['id']+'.f32';checks.append(f['input_sha256']+'  '+inp+'\n')
   for variant,nbhash in [('original',m['source_nb_sha256']),('candidate',m['nb_sha256'])]:
    sequence.append({**f,'run_id':f['id']+'-'+variant,'fixture_id':f['id'],'variant':variant,'nb':(shared+'/original.nb' if variant=='original' else tmp+'/candidate.nb'),'nb_sha256':nbhash,'input':inp,'local_reference':str(ref)})
 runner=R/'board_down_semantic.py'
 cfg={'binary':shared+'/tensor-probe','binary_sha256':PIN,'warmup':1,'repeats':2,'sequence':sequence,'scope':s['scope'],'fixture_status_sha256':sha(D/'status.json'),'source_sha256':{p.name:sha(p) for p in [Path(__file__),runner]},'input_bytes':total,'timing_claim':'two repeats for determinism, not percentile benchmark'}
 config=R/(RUN+'-config.json');config.write_text(json.dumps(cfg,indent=2)+'\n')
 with b.hostlock():
  check=m['source_nb_sha256']+'  '+shared+'/original.nb\n'+PIN+'  '+shared+'/tensor-probe\n'
  locked('sha256sum -c -',input=check.encode(),capture_output=True,check=True)
  p=locked('df -B1 --output=avail /dev/shm && cat /proc/meminfo && df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True)
  lines=p.stdout.splitlines();assert int(lines[1])>total+archive.stat().st_size+80*1024**2
  assert int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024>total+archive.stat().st_size+288*1024**2
  assert int(lines[-1])>1024**3+100*1024**2+2*1024**2
  locked('mkdir '+shlex.quote(remote)+' && mkdir '+shlex.quote(tmp),check=True);s.update(status='uploading',claimed=True);save()
  upload(Path(m['nb_path']),tmp+'/candidate.nb');upload(archive,tmp+'/inputs.tar.gz');upload(config,remote+'/config.json');upload(runner,remote+'/runner.py')
  check=m['nb_sha256']+'  '+tmp+'/candidate.nb\n'+sha(archive)+'  '+tmp+'/inputs.tar.gz\n'+sha(config)+'  '+remote+'/config.json\n'+sha(runner)+'  '+remote+'/runner.py\n'
  locked('sha256sum -c - && tar -xzf '+shlex.quote(tmp+'/inputs.tar.gz')+' -C '+shlex.quote(tmp),input=check.encode(),capture_output=True,check=True)
  locked('sha256sum -c -',input=''.join(checks).encode(),capture_output=True,check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 1800 python3 -u '+shlex.quote(remote+'/runner.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)
  script=cmd+' >'+shlex.quote(remote+'/process.log')+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(remote+'/exit')
  b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',timeout=20,check=True)
  s.update(status='running');save();deadline=time.monotonic()+1920
  while True:
   time.sleep(5);p=b.ssh('cat '+shlex.quote(remote+'/exit'),capture_output=True,text=True,timeout=20)
   if p.returncode==0:s['returncode']=int(p.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('no automatic restart')
  collect();meta=json.loads((R/'results'/RUN/'meta.json').read_text());assert s['returncode']==0 and meta['status']=='complete',meta
  locked('sha256sum -c -',input=''.join(checks).encode(),capture_output=True,check=True)
  # Remove only this run's verified scratch files after full evidence recovery.
  names=[tmp+'/'+f['id']+'.f32' for f in st['cases']]+[tmp+'/inputs.tar.gz',tmp+'/candidate.nb']
  locked('rm -- '+' '.join(map(shlex.quote,names)),check=True)
  s.update(status='complete',ended=time.time(),tmpfs_inputs_removed=True,tmpfs_results_preserved=output);save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time());save()
 if s.get('claimed') and not (R/'results'/RUN).exists():
  try:
   with b.hostlock():collect()
   s['partial_recovered']=True
  except BaseException:s['recovery_error']=traceback.format_exc()
  save()
 raise
