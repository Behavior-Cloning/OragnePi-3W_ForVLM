import pathlib,sys,subprocess,time,tarfile,shlex,json,hashlib,statistics,os
P=pathlib.Path(__file__).parent
R=P.parents[2]/'research/llama32-compat-20261023'
O=pathlib.Path('/opt/opi-lab/llama32-compat-20261023')
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/llama32-compat-20261023/zh-probe'
variants=[('attention-fp16-mlp-w8','227225b1ea4d8a44b508990b2602948d4930537dfdf5a0f8083b80a523826a8b',66878544),('all-fp16','031202e00e8609a07b4434700ba15f1323a4831421239787b0fee7500bdce36f',121813728)]
expected_ids=[0,1,2,4,8,14,100,101,102,110,111,112,120,121,122]
fixture_tar=O/'fixtures-zh.tar'
fixture_hash='4947b418a9af6923b1096145b8e22f2c20c1432adb7af5ce080cca8e96d5f573'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def locked(cmd,timeout=30):
 return b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=timeout,capture_output=True,text=True)
def board_python(code,timeout=30):
 return locked('python3 -c '+shlex.quote(code),timeout)
assert digest(fixture_tar)==fixture_hash
for name,sha,size in variants:
 f=O/name/'export_nbg_unify/network_binary.nb'
 assert f.stat().st_size==size and digest(f)==sha
result=[];cleanup=[]
with b.hostlock():
 locked('mkdir -p '+B,20).check_returncode()
 for f in [R/'layer_probe_zh.c',R/'board_probe.c',fixture_tar]:
  b.scp(str(f),b.HOST+':'+B+'/'+f.name,timeout=60).check_returncode()
 prep=locked('cd '+B+' && tar -xf fixtures-zh.tar && cc -O2 -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 layer_probe_zh.c -lNBGlinker -lVIPhal -lm -o layer_probe_zh',60)
 (P/'llama-final-probe-build.log').write_text(prep.stdout+prep.stderr);prep.check_returncode()
 for name,sha,size in variants:
  D=R/('board-results-'+name+'-zh');D.mkdir(exist_ok=True)
  if (D/'runner-exit.txt').exists():raise RuntimeError('Refusing to overwrite existing probe '+str(D))
  model=O/name/'export_nbg_unify/network_binary.nb'
  print('TRANSFER',name,size,flush=True)
  b.scp(str(model),b.HOST+':'+B+'/network.nb',timeout=240).check_returncode()
  check=locked('sha256sum '+B+'/network.nb',40);check.check_returncode();assert check.stdout.split()[0]==sha
  out='outputs-'+name;log='probe-'+name+'.log'
  cmd='cd '+B+' && mkdir -p '+out+' && timeout --kill-after=5s 120s ./layer_probe_zh network.nb fixtures '+out+' >'+log+' 2>&1'
  run=locked(cmd,145);(D/'runner-exit.txt').write_text(str(run.returncode)+'\n')
  (D/'controller-output.txt').write_text(run.stdout+run.stderr)
  locked('tar -cf '+B+'/results.tar -C '+B+' '+log+' '+out,30).check_returncode()
  b.scp(b.HOST+':'+B+'/results.tar',str(D/'results.tar'),timeout=45).check_returncode()
  with tarfile.open(D/'results.tar') as tf:tf.extractall(D,members=[m for m in tf.getmembers() if not m.issym() and not m.islnk()],filter='data')
  (D/log).rename(D/'probe.log');(D/out).rename(D/'outputs')
  rows=[]
  for line in (D/'probe.log').read_text(errors='replace').splitlines():
   if line.startswith('{'):
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
  fixtures=[x for x in rows if x.get('event')=='fixture'];closed=[x for x in rows if x.get('event')=='closed']
  ok=run.returncode==0 and [x['fixture_id'] for x in fixtures]==expected_ids and len(list((D/'outputs').glob('*.f32')))==15 and closed and closed[-1]['driver_released']==0
  info=dict(variant=name,model_sha256=sha,model_bytes=size,fixture_tar_sha256=fixture_hash,fixture_ids=expected_ids,source_sha256={f.name:digest(f) for f in [R/'layer_probe_zh.c',R/'board_probe.c']},result=str(D),runner_exit=run.returncode,execution_and_cleanup_complete=bool(ok),fixture_count=len(fixtures),median_run_ms=statistics.median(x['run_seconds'] for x in fixtures)*1000 if fixtures else None,worst_hidden_relative_l2=max((x['hidden_relative_l2'] for x in fixtures),default=None),closed=closed)
  (D/'source-and-result-manifest.json').write_text(json.dumps(info,indent=2));result.append(info)
  print(json.dumps(info),flush=True)
  rem=board_python('from pathlib import Path; p=Path('+repr(B+'/network.nb')+'); assert p.stat().st_size=='+str(size)+'; p.unlink(); print("removed NBG")');rem.check_returncode()
  cleanup.append(dict(path=B+'/network.nb',variant=name,bytes=size,sha256=sha,removed=True))
  if not ok:raise RuntimeError('Probe did not complete expected states: '+name)
 # Only remove this run's fixture staging and output copies after all raw evidence is on F.
 code='''
import pathlib,shutil,json,hashlib,os,time
base=pathlib.Path('/opt/opi-vlm/llama32-compat-20261023/zh-probe')
removed=[]
for name in ['fixtures','fixtures-zh.tar','outputs-attention-fp16-mlp-w8','outputs-all-fp16','results.tar']:
 p=base/name
 if p.exists():
  size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) if p.is_dir() else p.stat().st_size
  shutil.rmtree(p) if p.is_dir() else p.unlink()
  removed.append({'path':str(p),'bytes':size})
active=[]
for d in pathlib.Path('/proc').iterdir():
 if d.name.isdigit():
  try:
   exe=os.readlink(d/'exe')
   if exe.startswith('/opt/opi-vlm/vlm-new-models-20261023/') and 'gemma' in exe:active.append({'pid':d.name,'exe':exe})
  except OSError:pass
assert not active,active
p=pathlib.Path('/opt/opi-vlm/vlm-new-models-20261023/paligemma2-3b-mix-224-sfp.sbs')
assert p.stat().st_size==4045406208
h=hashlib.sha256()
with p.open('rb') as f:
 for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
sha=h.hexdigest();assert sha=='c3bea2caf002da77624d807b79f26519455fc8326eca99442b21b3e270c44b73'
p.unlink();removed.append({'path':str(p),'bytes':4045406208,'sha256':sha,'F_original_preserved':True})
u=shutil.disk_usage('/opt/opi-vlm')
print(json.dumps({'removed':removed,'active_task_gemma_processes':active,'disk_free_bytes':u.free,'disk_total_bytes':u.total,'timestamp_unix':time.time()}))
'''
 clean=board_python(code,240);(P/'final-board-cleanup-output.log').write_text(clean.stdout+clean.stderr);clean.check_returncode()
 detail=json.loads(clean.stdout.strip().splitlines()[-1]);detail['nbg_staging_removed']=cleanup
 (P/'final-board-staging-cleanup.json').write_text(json.dumps(detail,indent=2))
 (P/'llama-final-probes-status.json').write_text(json.dumps(dict(results=result,cleanup=detail),indent=2))
 print('BOARD_CLEANUP',json.dumps(detail),flush=True)
# No further board calls after the host lock is released.
print('READY_COMPACT: final two probes returned, driver released, board hostlock released, own controller exiting; parent keeper is untouched.',flush=True)
