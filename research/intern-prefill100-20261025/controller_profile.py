#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,sys,tarfile,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
R=Path(__file__).resolve().parent;name='single-block-profile-r1';status=R/(name+'-controller.json');assert not status.exists()
source=json.loads((R/'single-block-c512-r1-config.json').read_text());rows={x['run_id']:x for x in source['sequence']}
remote='/opt/opi-vlm/intern-prefill100-20261025/runs/'+name;s={'status':'starting','started':time.time(),'cases':[]}
def save():
 p=status.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(status)
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',30),**kw)
save()
try:
 with b.hostlock():
  locked('mkdir '+shlex.quote(remote),check=True)
  verify={source['binary']:source['binary_sha256']}
  for key in ['c512-m16-l0','c512-m16-l2','c512-m32-l0','c512-m32-l2']:
   row=rows[key];verify[row['nb']]=row['nb_sha256'];verify[row['input']]=row['input_sha256']
  check=''.join(h+'  '+p+'\n' for p,h in verify.items()).encode();locked('sha256sum -c -',input=check,capture_output=True,check=True)
  for mode in ['io','npd']:
   for key in ['c512-m16-l0','c512-m16-l2','c512-m32-l0','c512-m32-l2']:
    row=rows[key];case=key+'-'+mode;folder=remote+'/'+case
    env='OPI_PROBE_IO_PROFILE=1 OPI_PROBE_WARMUP=3' if mode=='io' else 'OPI_PROBE_NPD=1 OPI_PROBE_WARMUP=0'
    repeats='5' if mode=='io' else '1'
    cmd='mkdir '+shlex.quote(folder)+' && cd '+shlex.quote(folder)+' && timeout -k 5 90 env '+env+' taskset -c 4,5 '+shlex.quote(source['binary'])+' '+shlex.quote(row['nb'])+' '+shlex.quote(row['input'])+' output.f32 '+repeats+' result.json >process.log 2>&1'
    s.update(status='running',active_case=case);save();start=time.time()
    z=locked(cmd,timeout=105,capture_output=True)
    entry={'case':case,'mode':mode,'returncode':z.returncode,'seconds':time.time()-start,'timing_eligible':False};s['cases'].append(entry);save()
    packet=locked('tar -czf - -C '+shlex.quote(folder)+' .',timeout=45,capture_output=True,check=True)
    archive=R/(case+'.tar.gz');archive.write_bytes(packet.stdout);dest=R/'results'/name/case;dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as tar:
     for member in tar.getmembers():
      q=(dest/member.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents;assert not member.issym() and not member.islnk()
     tar.extractall(dest)
    result=dest/'result.json';entry['result']=json.loads(result.read_text()) if result.exists() else None
    if entry['result']:assert entry['result']['released_bytes']==0
    entry['artifacts']=[{'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.rglob('*') if p.is_file()];save()
    if z.returncode:raise RuntimeError('diagnostic failed: '+case)
  s.update(status='complete',ended=time.time());save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
