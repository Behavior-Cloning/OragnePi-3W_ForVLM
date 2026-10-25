#!/usr/bin/env python3
"""Independent M256 diagnostics only; profile timings are not pure latency."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess,threading,time,traceback
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
(out/'config.json').write_text(json.dumps(cfg,indent=2));stop=threading.Event();err=[];state={'case':None};events=(out/'events.jsonl').open('w',buffering=1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(kind,**kw):events.write(json.dumps({'event':kind,'at':time.monotonic(),**kw})+'\n')
def snapshot():
 mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
 temps={p.parent.name:int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
 return {'at':time.monotonic(),'case':state['case'],'temperatures_mC':temps,'memory':mem,'vmstat':Path('/proc/vmstat').read_text(),'stat':Path('/proc/stat').read_text(),'loadavg':Path('/proc/loadavg').read_text(),'frequency_kHz':{p.parts[-3]:int(p.read_text()) for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')},'disk_free':shutil.disk_usage(out).free,'driver':Path('/sys/kernel/debug/viplite/mem_profile').read_text()}
def monitor():
 try:
  with (out/'resources.jsonl').open('w',buffering=1) as f:
   while not stop.wait(.25):
    row=snapshot();f.write(json.dumps(row)+'\n')
    if max(row['temperatures_mC'].values())>=78000:raise RuntimeError('78C guard')
 except BaseException:err.append(traceback.format_exc());stop.set()
assert sha(cfg['binary'])==cfg['binary_sha256']
assert cfg['diagnostics_only'] is True and len(cfg['sequence'])==2
assert [r['mode'] for r in cfg['sequence']]==['io','npd']
(out/'pre-health.json').write_text(json.dumps(snapshot(),indent=2))
th=threading.Thread(target=monitor,daemon=True);th.start();complete=False;proc=None
try:
 for row in cfg['sequence']:
  state['case']=row['run_id'];assert not err
  mode=row['mode'];warm=3 if mode=='io' else 0;reps=5 if mode=='io' else 1
  folder=out/row['run_id'];folder.mkdir()
  assert sha(row['nb'])==row['nb_sha256'] and sha(row['input'])==row['input_sha256']
  deadline=time.monotonic()+180
  while max(snapshot()['temperatures_mC'].values())>=65000:
   if time.monotonic()>deadline:raise RuntimeError('cooldown timeout')
   if stop.wait(2):raise RuntimeError(str(err))
  env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')};env['OPI_PROBE_WARMUP']=str(warm);env['OPI_PROBE_IO_PROFILE' if mode=='io' else 'OPI_PROBE_NPD']='1'
  result=folder/'result.json';dump=folder/'output.f32'
  cmd=['taskset','-c','4,5',cfg['binary'],row['nb'],row['input'],str(dump),str(reps),str(result)]
  start=time.monotonic();emit('start',case=row['run_id'],query_tokens=row['query_tokens'],first_layer=row['first_layer'],command=cmd)
  with (folder/'process.log').open('w') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=folder)
   while proc.poll() is None:
    if err or time.monotonic()-start>180:
     proc.kill();proc.wait();raise RuntimeError(str(err or 'per probe timeout'))
    time.sleep(.1)
  assert proc.returncode==0, (row['run_id'],proc.returncode)
  data=json.loads(result.read_text());assert data['released_bytes']==0 and data['repeats']==reps and data['warmup_repeats']==warm
  assert data['capacity']==512 and data['query_rows']==256 and data['fusion_layers']==2 and data['scope']=='two_layer_probe'
  npds=list(folder.glob('*.npd'))
  if mode=='npd':
   assert len(npds)==1 and npds[0].stat().st_size>0
   with npds[0].open('rb') as nf: assert nf.read(4)==b'VNPD'
  else: assert not npds
  assert data['npd_enabled']==(mode=='npd') and len(data['api_seconds'])==reps
  assert data['io_profile_enabled']==(mode=='io') and data['resident_copies']==0
  assert all(math.isfinite(x) and x>0 for x in data['api_seconds'])
  assert data['mismatched_elements']==0 and data['differing_repeats']==0
  expected=4*row['query_tokens']*(896+4*128);assert dump.stat().st_size==expected
  emit('result',case=row['run_id'],query_tokens=row['query_tokens'],first_layer=row['first_layer'],metrics=data,output_sha256=sha(dump),output_bytes=expected,wall_seconds=time.monotonic()-start,scope='M256 independent diagnostic; never pure timing or full model',timing_eligible=False,mode=mode,artifacts=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in folder.iterdir() if f.is_file()])
 complete=True
except BaseException:emit('error',traceback=traceback.format_exc());raise
finally:
 if proc and proc.poll() is None:proc.kill();proc.wait()
 stop.set();th.join(5)
 if th.is_alive():err.append('monitor thread failed to stop')
 (out/'post-health.json').write_text(json.dumps(snapshot(),indent=2));meta={'status':'complete' if complete and not err else 'failed','monitor_errors':err,'scope':'single-block profile diagnostics only; not pure timing, full model or TTFT; no audio/RAG residency'};(out/'meta.json').write_text(json.dumps(meta,indent=2));emit('complete' if meta['status']=='complete' else 'failed',**meta);events.close()
