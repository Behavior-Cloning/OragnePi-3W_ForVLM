import subprocess,time,urllib.request,json,pathlib,threading,os,hashlib
P=pathlib.Path(__file__).parent; out=open(P/'llama-cached-results.jsonl','w')
def emit(d):out.write(json.dumps(d,ensure_ascii=False)+'\n');out.flush()
model=P/'FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf'; samples=[];stop=threading.Event()
def monitor(pid):
 while not stop.wait(.25):
  try:
   status=dict(x.split(':',1) for x in pathlib.Path('/proc/'+str(pid)+'/status').read_text().splitlines() if ':' in x)
   smaps=dict(x.split(':',1) for x in pathlib.Path('/proc/'+str(pid)+'/smaps_rollup').read_text().splitlines() if ':' in x)
   temps=[float(x.read_text())/1000 for x in pathlib.Path('/sys/class/thermal').glob('thermal_zone*/temp')]
   st=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split();mem=dict(x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines())
   samples.append(dict(t=time.monotonic(),rss_kb=int(status['VmRSS'].split()[0]),pss_kb=int(smaps['Pss'].split()[0]),temp_max=max(temps),cpu_ticks=int(st[13])+int(st[14]),available_kb=int(mem['MemAvailable'].split()[0])))
  except Exception:pass
cmd=['taskset','-c','6,7','/opt/opi-vlm/bin/llama-server','-m',str(model),'-t','2','-tb','2','-c','1024','-np','1','-ngl','0','--host','127.0.0.1','--port','8086']
log=open(P/'llama-cached-server.log','w');start=time.monotonic();srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);mon=threading.Thread(target=monitor,args=(srv.pid,),daemon=True);mon.start()
try:
 for _ in range(180):
  if srv.poll() is not None:raise RuntimeError('server exited '+str(srv.returncode))
  try:
   with urllib.request.urlopen('http://127.0.0.1:8086/health',timeout=1) as r:
    if r.status==200:break
  except Exception:time.sleep(1)
 else:raise TimeoutError('server load')
 emit(dict(event='ready',load_s=time.monotonic()-start,cmd=cmd,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
 cases=[('greeting','我下班回家了，有点累，用一句简短的话欢迎我。'),('water','我刚运动回来，请用一句话提醒我休息和喝水。'),('quiet','家里有人睡觉，请用一句话告诉我你会保持安静。')]
 for n in range(7):
  cid,user=cases[n%3];prompt='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n你是家庭陪伴机器人，请用简短中文回答。<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'+user+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  req=urllib.request.Request('http://127.0.0.1:8086/completion',data=json.dumps(dict(prompt=prompt,n_predict=32,temperature=0,seed=42,stream=True,cache_prompt=True)).encode(),headers={'Content-Type':'application/json'})
  t0=time.monotonic();first=None;content='';final={}
  with urllib.request.urlopen(req,timeout=90) as r:
   for line in r:
    if not line.startswith(b'data:'):continue
    payload=line[5:].strip()
    if payload==b'[DONE]':continue
    d=json.loads(payload)
    if d.get('content'):
     if first is None:first=time.monotonic()
     content+=d['content']
    if d.get('stop'):final=d
  emit(dict(event='measure',warmup=n==0,case=cid,request=n,prompt=prompt,ttft_s=None if first is None else first-t0,total_s=time.monotonic()-t0,text=content,timings=final.get('timings'),tokens_predicted=final.get('tokens_predicted'),tokens_evaluated=final.get('tokens_evaluated'),cache_prompt=True))
 emit(dict(event='complete'))
except Exception as e:emit(dict(event='error',error=repr(e)));raise
finally:
 srv.terminate()
 try:srv.wait(timeout=10)
 except subprocess.TimeoutExpired:srv.kill();srv.wait()
 stop.set();mon.join(timeout=2);(P/'llama-cached-telemetry.json').write_text(json.dumps(samples));out.close()
