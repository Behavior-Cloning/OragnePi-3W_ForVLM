"""CPU-only offline ASR comparison; paced mode replays public WAV in wall time."""
import argparse,json,os,resource,time,wave
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sherpa_onnx as s
ROOT=Path(__file__).resolve().parent
OLD=Path('/opt/opi-vlm/audio-lab-20261018')
NAMES={'stream':'sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01','offline':'sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16','para':'sherpa-onnx-paraformer-zh-small-2024-03-09','qwen':'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25'}
def stats():
 out={'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'temperature_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())}
 for f,keys in [('/proc/self/status',('VmRSS:','Threads:')),('/proc/self/smaps_rollup',('Pss:',))]:
  for l in Path(f).read_text().splitlines():
   if l.startswith(keys):out[l.split(':')[0]]=int(l.split()[1])
 return out
def model(k,threads=1):
 if k=='sense':return s.OfflineRecognizer.from_sense_voice(model=str(OLD/'models/sensevoice/model.int8.onnx'),tokens=str(OLD/'models/sensevoice/tokens.txt'),num_threads=threads,use_itn=False)
 d=ROOT/'models'/NAMES[k];kw=dict(tokens=str(d/'tokens.txt'),num_threads=threads)
 if k=='stream':return s.OnlineRecognizer.from_zipformer2_ctc(model=str(d/'model.int8.onnx'),enable_endpoint_detection=True,rule1_min_trailing_silence=2.0,rule2_min_trailing_silence=.5,rule3_min_utterance_length=30,**kw)
 if k=='offline':return s.OfflineRecognizer.from_zipformer_ctc(model=str(d/'model.int8.onnx'),**kw)
 if k=='para':return s.OfflineRecognizer.from_paraformer(paraformer=str(d/'model.int8.onnx'),**kw)
 raise ValueError(k)
def readwav(p):
 with wave.open(str(p)) as f:
  assert f.getsampwidth()==2 and f.getnchannels()==1 and f.getframerate()==16000
  return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768

def infer(k,m,x,paced=False):
 st=m.create_stream();events=[];endpoints=[];t=time.perf_counter();cpu=time.process_time()
 if k!='stream':
  st.accept_waveform(16000,x);m.decode_stream(st);return {'text':st.result.text,'seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu}
 # 80 ms arrivals. Record natural endpoint events; do not reset mid-clip,
 # so the final transcript remains comparable to an utterance recognizer.
 last='';decode_times=[];duration=len(x)/16000;chunks=1280
 y=np.concatenate([x,np.zeros(32000,dtype=np.float32)])
 for i in range(0,len(y),chunks):
  chunk=y[i:i+chunks];arrival=(i+len(chunk))/16000
  if paced:time.sleep(max(0,t+arrival-time.perf_counter()))
  dt=time.perf_counter();st.accept_waveform(16000,chunk)
  while m.is_ready(st):m.decode_stream(st)
  decode_times.append(time.perf_counter()-dt);txt=m.get_result(st)
  if txt!=last:events.append({'at_seconds':time.perf_counter()-t,'audio_seconds':arrival,'text':txt});last=txt
  if m.is_endpoint(st):endpoints.append({'at_seconds':time.perf_counter()-t,'audio_seconds':arrival,'text':txt})
 before_flush=m.get_result(st);st.input_finished()
 while m.is_ready(st):m.decode_stream(st)
 final=m.get_result(st);end=time.perf_counter()-t
 natural=next((e for e in endpoints if e['audio_seconds']>=duration),None)
 stable=next((e for j,e in enumerate(events) if e['text']==final and all(z['text']==final for z in events[j:])),None)
 return {'text':final,'seconds':end,'cpu_seconds':time.process_time()-cpu,'paced':paced,'events':events,'endpoints':endpoints,'natural_endpoint_after_clip':natural,'before_flush_text':before_flush,'flush_changed_text':before_flush!=final,'first_partial_seconds':events[0]['at_seconds'] if events else None,'stable_final_seconds':stable['at_seconds'] if stable else None,'clip_end_to_endpoint_seconds':natural['at_seconds']-duration if natural and paced else None,'clip_end_to_stable_seconds':stable['at_seconds']-duration if stable and paced else None,'decode_compute_seconds':sum(decode_times),'chunk_compute_max_ms':max(decode_times)*1000,'forced_finish_seconds':end,'chunk_ms':80,'added_silence_seconds':2}

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--paced',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--ready');p.add_argument('--go');p.add_argument('--stop');a=p.parse_args()
 corpus=json.loads((ROOT/'corpus/manifest.json').read_text());samples=corpus['samples'];samples=samples[:a.limit] if a.limit else samples
 t=time.perf_counter();m=model(a.kind);load=time.perf_counter()-t;loaded=stats();runs=[]
 # Explicit unscored warm-up, separate from resident inference timings.
 warm=infer(a.kind,m,readwav(ROOT/'corpus'/samples[0]['file']))
 if a.ready:Path(a.ready).write_text(str(os.getpid()))
 if a.go:
  deadline=time.monotonic()+600
  while not Path(a.go).exists():
   if time.monotonic()>deadline:raise TimeoutError('go')
   time.sleep(.02)
 iteration=0
 while True:
  row=samples[iteration%len(samples)];x=readwav(ROOT/'corpus'/row['file']);start=time.perf_counter()
  r=infer(a.kind,m,x,a.paced);r.update(id=row['id'],reference=row['text'],duration_seconds=len(x)/16000,start_monotonic=start,end_monotonic=time.perf_counter(),resources=stats(),iteration=iteration);r['rtf']=r['seconds']/r['duration_seconds'];runs.append(r)
  print(a.kind,row['id'],round(r['seconds'],3),r['text'],flush=True);iteration+=1
  if a.stop:
   if Path(a.stop).exists():break
   if iteration>=10000:raise RuntimeError('stop missing')
  elif iteration>=len(samples)*a.repeats:break
 out={'kind':a.kind,'status':'complete','affinity':sorted(os.sched_getaffinity(0)),'threads':1,'runtime':s.__version__,'load_seconds':load,'load_cache_state':'OS cache not flushed','loaded':loaded,'warmup':warm,'final':stats(),'runs':runs,'timing_scope':'features + recognition API, resident model; no microphone/network; stream includes 2 s synthetic tail, paced includes audio playback time'}
 Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
