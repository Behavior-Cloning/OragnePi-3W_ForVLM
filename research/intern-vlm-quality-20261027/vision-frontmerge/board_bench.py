from pathlib import Path
import subprocess,json,time,threading,hashlib
P=Path(__file__).resolve().parent;O=P/'results';O.mkdir(exist_ok=False)
stop=threading.Event();child=[None]
def monitor():
 with (O/'resources.jsonl').open('w') as f:
  while not stop.is_set():
   x={'time':time.monotonic(),'meminfo':Path('/proc/meminfo').read_text(),'cpu':Path('/proc/stat').read_text().splitlines()[0],'temperatures':{str(p):p.read_text().strip() for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}}
   if child[0]:
    try:x['process_status']=Path('/proc/'+str(child[0])+'/status').read_text()
    except FileNotFoundError:pass
   f.write(json.dumps(x)+'\n');f.flush();stop.wait(.5)
th=threading.Thread(target=monitor);th.start()
s={'scope':'CPU vision encoder+projector only; no LLM/audio/RAG; preprocessing excluded; not VQA validation','phases':[],'start':time.time()}
try:
 for name,mode in [('native-before','native'),('front','front'),('late','late'),('native-after','native')]:
  req=[{'id':name+'-'+str(i),'side':448,'threads':2,'input':str(P/(im+'.f32')),'output':str(O/(name+'-'+str(i)+'.f32'))} for i,im in enumerate(['image1','image1','image2','image1'])]+[{'op':'close'}]
  with (O/(name+'.stderr')).open('w') as err:
   z=subprocess.Popen(['taskset','-c','6,7',str(P/'vision-worker'),'/opt/opi-vlm/models/internvl3-1b/mmproj-InternVL3-1B-Instruct-f16.gguf',mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True);child[0]=z.pid
   try:stdout,_=z.communicate(''.join(json.dumps(q)+'\n' for q in req),timeout=240)
   except subprocess.TimeoutExpired:z.kill();stdout,_=z.communicate();(O/(name+'.jsonl')).write_text(stdout);raise
  child[0]=None;(O/(name+'.jsonl')).write_text(stdout);ev=[json.loads(x) for x in stdout.splitlines() if x.startswith('{')]
  state={'phase':name,'mode':mode,'returncode':z.returncode,'events':ev};s['phases'].append(state);(O/'summary.json').write_text(json.dumps(s,indent=2))
  assert z.returncode==0 and len([x for x in ev if x.get('event')=='encoded'])==4 and ev[-1]['event']=='closed',state
 s['state']='completed'
except Exception as e:s['state']='failed';s['error']=str(e)
finally:
 stop.set();th.join();s['end']=time.time();s['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.f32')};(O/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps({'state':s['state'],'phases':len(s['phases'])}))
