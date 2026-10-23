"""Host fault-injection test of board runner cleanup; no board or model invocation."""
from pathlib import Path
import subprocess,signal,time,json,tempfile,hashlib
R=Path(__file__).resolve().parent;B=Path('/opt/opi-lab/llama32-full-npu-20261023/q4-source');rows=[]
for case in ['term','thermal','memory']:
 with tempfile.TemporaryDirectory(prefix='llama-guard-') as td:
  d=Path(td);code=(R/'board_run.py').read_text()
  if case=='thermal':code=code.replace("temps={p.parent.name:int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}","temps={'injected':79.0}")
  if case=='memory':code=code.replace('processes=[]',"mem['MemAvailable_kB']=128*1024;processes=[]")
  (d/'runner.py').write_text(code)
  (d/'resident_worker.py').write_text("from pathlib import Path\nimport time\nclass ResidentWorker:\n def __init__(self,*a,**k):self.ready={'capacity':256}\n def generate(self,*a,**k):\n  time.sleep(30)\n  yield {}\n def abort(self):Path(__file__).with_name('aborted').write_text('yes')\n")
  assets=d/'assets';assets.mkdir();(assets/'inputs.json').write_bytes((B/'inputs.json').read_bytes());(assets/'reference.json').write_bytes((B/'oracle/reference.json').read_bytes())
  with (d/'stderr').open('w') as log:
   p=subprocess.Popen(['python3',str(d/'runner.py'),'--graphs',str(d),'--assets',str(assets),'--output',str(d/'output')],stdout=log,stderr=subprocess.STDOUT)
   if case=='term':time.sleep(.8);p.send_signal(signal.SIGTERM)
   rc=p.wait(timeout=5)
  events=[json.loads(x) for x in (d/'output/events.jsonl').read_text().splitlines()];tele=json.loads((d/'output/telemetry.json').read_text());assert rc!=0 and (d/'aborted').exists() and any(x['event']=='cleanup' for x in events) and any(x['event']=='error' for x in events)
  if case!='term':assert tele['guards']
  rows.append(dict(case=case,returncode=rc,own_worker_abort=True,telemetry_saved=True,cleanup_event=True,guards=tele['guards']))
(R/'guard-test-results.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256((R/'board_run.py').read_bytes()).hexdigest(),tests=rows,scope='fault-injected fake worker; no board or NPU'),indent=2));print(json.dumps(rows))
