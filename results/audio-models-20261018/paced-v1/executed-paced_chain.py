"""75s file-replay streaming prototype with bounded queues; no microphone or service changes."""
import atexit,hashlib,json,multiprocessing as mp,os,queue,statistics,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
import audio_worker as w
import board_trials as bt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'paced-v1'
def worker(kind,inbox,ready):
 os.sched_setaffinity(0,{2,3} if kind=='asr' else {4,5});m=w.model(kind,1);ready.set();rows=[]
 while True:
  item=inbox.get()
  if item is None:break
  index,submitted,x=item;start=time.monotonic();value,_=w.infer(kind,m,x)
  if kind=='speaker':value={k:v for k,v in value.items() if k!='embedding'}
  rows.append({'segment':index,'queued_at':submitted,'start':start,'end':time.monotonic(),'queue_delay_seconds':start-submitted,'audio_seconds':len(x)/16000,'output':value})
 (OUT/(kind+'.json')).write_text(json.dumps({'kind':kind,'runs':rows,'resources':w.stats(),'cpu_affinity':sorted(os.sched_getaffinity(0))},ensure_ascii=False,indent=2))
def main():
 OUT.mkdir(exist_ok=False);os.sched_setaffinity(0,{0,1});mp.set_start_method('spawn');kinds=['asr','speaker'];qs={k:mp.Queue(2) for k in kinds};events={k:mp.Event() for k in kinds};children={k:mp.Process(target=worker,args=(k,qs[k],events[k])) for k in kinds}
 for c in children.values():c.start()
 atexit.register(lambda:[c.terminate() for c in children.values() if c.is_alive()])
 for e in events.values():assert e.wait(120)
 den=w.model('denoise',1);vad=w.model('vad',1);frame=den.frame_shift_in_samples;assert frame==256
 clean=w.read_wav(ROOT/'audio/zh.wav');english=w.read_wav(ROOT/'audio/en.wav');rng=np.random.default_rng(20261018);noise=rng.standard_normal(len(clean)).astype(np.float32);noise*=np.sqrt(np.mean(clean**2)/np.mean(noise**2));noisy=(clean+noise)/max(1.,float(np.max(np.abs(clean+noise)))/.99)
 cycle=np.concatenate([clean,np.zeros(16000,np.float32),noisy,np.zeros(16000,np.float32),english,np.zeros(48000,np.float32)])
 silence_probe,_=w.infer('vad',vad,np.zeros(48000,np.float32));assert not silence_probe['segments'];vad.reset()
 duration=75.;signal=np.tile(cycle,int(np.ceil(duration*16000/len(cycle))))[:int(duration*16000)]
 with (OUT/'thermal.log').open('w') as log:subprocess.run([sys.executable,'thermal_condition.py','--min','55000','--max','64999'],cwd=bt.NPU,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=330)
 pids=[os.getpid(),*[c.pid for c in children.values()]];monitor=[];done=threading.Event();mt=threading.Thread(target=bt.monitor,args=(done,pids,monitor));mt.start();llms=[];errors=[];frames=[];segments=[];dropped={k:0 for k in kinds};buffer=np.zeros(0,np.float32);start=time.monotonic()
 def llm_work():
  try:
   for dynamic,delay in [(64,10),(256,40)]:
    while time.monotonic()<start+delay:time.sleep(.05)
    env={k:v for k,v in bt.ENV.items() if not k.startswith('OPI_')};env.update(OPI_PERFORMANCE='1',OPI_DECODER_GROUP='12',OPI_FAST_GREEDY='1',OPI_GROUP_HEAD='1',OPI_PREFILL_TOKENS='16',OPI_CACHED_PREFIX='64',OPI_MAX_START_TEMP_MC='70000',OPI_FEEDBACK_FILE=str(bt.NPU/(bt.PREFIX+'-feedback.i32')))
    prompt=Path('/opt/opi-vlm/short-prefix-d64-20261018/p64-d64.i32') if dynamic==64 else Path('/opt/opi-vlm/short-prefix-20261018/p64-d256.i32')
    output=OUT/f'llm-d{dynamic}.json';cmd=['taskset','-c','6,7','./'+bt.PREFIX+'-full_model','.','24','1280','151674','embeddings.f32',str(prompt),'128',str(output),str(OUT/'unused.f32'),'151645']
    with (OUT/f'llm-d{dynamic}.log').open('w') as log:
     c=subprocess.Popen(cmd,cwd=bt.NPU,env=env,stdout=log,stderr=subprocess.STDOUT);pids.append(c.pid)
     try:assert c.wait(timeout=120)==0
     finally:
      if c.poll() is None:c.kill();c.wait()
    r=json.loads(output.read_text());assert r['generated_tokens']==128 and r['driver_after_release_bytes']==0 and r['head_on_npu'];llms.append(r)
  except BaseException as e:errors.append(repr(e))
 lt=threading.Thread(target=llm_work);lt.start()
 def drain(forced=False):
  while not vad.empty():
   seg=vad.front;x=np.asarray(seg.samples,np.float32).copy();index=len(segments);submitted=time.monotonic();segments.append({'index':index,'start_sample':seg.start,'samples':len(x),'emitted_monotonic':submitted,'vad_segment_end_due':start+(seg.start+len(x))/16000,'forced_at_end':forced})
   for k in kinds:
    try:qs[k].put_nowait((index,submitted,x))
    except queue.Full:dropped[k]+=1
   vad.pop()
 try:
  for i in range(0,len(signal),frame):
   due=start+(i+frame)/16000;time.sleep(max(0,due-time.monotonic()));t=time.monotonic();chunk=signal[i:i+frame];y=den(chunk,16000);buffer=np.concatenate([buffer,np.asarray(y.samples,np.float32)])
   while len(buffer)>=512:vad.accept_waveform(buffer[:512]);buffer=buffer[512:];drain()
   end=time.monotonic();frames.append({'due':due,'start':t,'end':end,'processing_ms':(end-t)*1000,'start_lag_ms':max(0,t-due)*1000,'completion_lag_ms':max(0,end-due)*1000})
  tail=np.concatenate([buffer,np.asarray(den.flush().samples,np.float32)])
  for i in range(0,len(tail),512):vad.accept_waveform(np.pad(tail[i:i+512],(0,max(0,512-len(tail[i:i+512])))));drain()
  vad.flush();drain(True)
  for q in qs.values():q.put(None,timeout=30)
  for c in children.values():c.join(120);assert c.exitcode==0
  lt.join(150);assert not lt.is_alive() and not errors,errors
 finally:
  for c in children.values():
   if c.is_alive():c.terminate();c.join()
  done.set();mt.join()
 summary={'status':'complete','scope':'75s 16kHz file replay at 16ms capture cadence, simulated capture, no microphone','duration_seconds':duration,'silence_gate':{'input_seconds':3,'vad_segments':silence_probe['segments'],'asr_calls':0,'memory_writes':0},'start_monotonic':start,'frames':len(frames),'segments':segments,'queue_capacity_per_worker':2,'dropped_segments':dropped,'p50_frame_ms':statistics.median(x['processing_ms'] for x in frames),'p95_frame_ms':w.quantile([x['processing_ms'] for x in frames],.95),'max_frame_ms':max(x['processing_ms'] for x in frames),'max_start_lag_ms':max(x['start_lag_ms'] for x in frames),'last_start_lag_ms':frames[-1]['start_lag_ms'],'frames_processing_over_16ms':sum(x['processing_ms']>16 for x in frames),'llm':llms,'resources':w.stats()}
 summary['backend']={}
 for kind in kinds:
  d=json.loads((OUT/(kind+'.json')).read_text());rs=d['runs'];summary['backend'][kind]={'calls':len(rs),'segment_ids':[r['segment'] for r in rs],'queue_wait_p50_seconds':statistics.median(r['queue_delay_seconds'] for r in rs) if rs else None,'queue_wait_max_seconds':max((r['queue_delay_seconds'] for r in rs),default=0),'vad_emission_to_result_p50_seconds':statistics.median(r['end']-r['queued_at'] for r in rs) if rs else None,'vad_segment_end_to_result_p50_seconds':statistics.median(r['end']-segments[r['segment']]['vad_segment_end_due'] for r in rs) if rs else None,'latency_reference':'VAD reported segment end, not human annotated utterance end'}
 summary['backend_segment_ids_match']=summary['backend']['asr']['segment_ids']==summary['backend']['speaker']['segment_ids']
 for name,data in [('summary',summary),('frames',frames),('system-resources',monitor)]:
  (OUT/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k not in ['llm','segments']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
