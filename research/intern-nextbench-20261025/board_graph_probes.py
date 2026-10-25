import os,sys,json,time,hashlib,subprocess,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=False);spec=json.loads((root/'spec.json').read_text());events=[]
def temp():return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
def save(): (out/'status.json').write_text(json.dumps({'events':events,'at':time.time()},indent=2))
try:
 for f,h in spec['hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h, f
 for q in spec['jobs']:
  assert temp()<78000
  dest=out/q['name'];dest.mkdir(parents=True)
  command=['env','OPI_PROBE_WARMUP=3','taskset','-c','4,5',q['probe'],q['graph'],q['input'],str(dest/'output.f32'),str(q['repeats']),str(dest/'timing.json')]
  t=time.time();start=temp()
  with (dest/'probe.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=75)
  row={'name':q['name'],'returncode':r.returncode,'elapsed_s':time.time()-t,'temperature_start_mC':start,'temperature_end_mC':temp()};events.append(row);save();assert r.returncode==0,row
  result=json.loads((dest/'timing.json').read_text());assert result.get('released_bytes')==0,result
 (out/'complete.json').write_text(json.dumps({'status':'complete','count':len(events)}))
except BaseException:
 (out/'failure.txt').write_text(traceback.format_exc());raise
