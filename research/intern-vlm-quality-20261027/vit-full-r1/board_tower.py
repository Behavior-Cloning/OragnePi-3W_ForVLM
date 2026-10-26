from pathlib import Path
import subprocess,threading,os,json,time,hashlib,re
P=Path(__file__).resolve().parent;O=P/'results';O.mkdir(exist_ok=False);cfg=json.loads((P/'config.json').read_text())
for n,h in cfg['sha256'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
stop=threading.Event();errors=[];child=[None]
def monitor():
 with (O/'resources.jsonl').open('w',buffering=1) as f:
  while not stop.wait(.5):
   mem=Path('/proc/meminfo').read_text();temps={str(p):int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')};record={'at':time.monotonic(),'meminfo':mem,'temperatures':temps,'cpu':Path('/proc/stat').read_text().splitlines()[0]}
   if child[0]:
    try:record['process_status']=Path('/proc',str(child[0]),'status').read_text()
    except FileNotFoundError:pass
   f.write(json.dumps(record)+'\n')
   if int(re.search(r'MemAvailable:\s*(\d+)',mem).group(1))<96*1024 or max(temps.values())>=78000:errors.append('memory/temperature guard');return
th=threading.Thread(target=monitor,daemon=True);th.start();rc=None;proc=None
try:
 with (O/'stack.log').open('w') as log:
  env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')}
  proc=subprocess.Popen(['taskset','-c','7',str(P/'tower-probe'),str(P/'graphs'),str(P),str(O)],env=env,stdout=log,stderr=subprocess.STDOUT);child[0]=proc.pid;start=time.monotonic()
  while proc.poll() is None:
   if errors or time.monotonic()-start>300:proc.kill();proc.wait();raise RuntimeError(str(errors or ['timeout']))
   time.sleep(.2)
  rc=proc.returncode;assert rc==0,rc
finally:
 if proc and proc.poll() is None:proc.kill();proc.wait()
 stop.set();th.join(3);(O/'status.json').write_text(json.dumps({'returncode':rc,'monitor_errors':errors},indent=2)+'\n')
