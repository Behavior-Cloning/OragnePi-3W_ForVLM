"""ARM64 CPU inference and resource measurement on fixed public audio.
Offline replay timings exclude capture duration, transport and endpoint waiting.
"""
import argparse,hashlib,json,math,os,resource,statistics,time,wave
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sherpa_onnx as s
ROOT=Path(__file__).resolve().parent
ASR=ROOT/'models/sensevoice'
def read_wav(path):
 with wave.open(str(path),'rb') as f:
  assert f.getsampwidth()==2
  rate=f.getframerate();x=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
  if f.getnchannels()>1:x=x.reshape(-1,f.getnchannels())[:,0]
 assert rate==16000,(path,rate)
 return np.ascontiguousarray(x)
def stats():
 data={}
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith(('VmRSS:','VmHWM:','Threads:')):data[line.split(':')[0]]=int(line.split()[1])
 try:
  for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
   if line.startswith('Pss:'):data['Pss_kib']=int(line.split()[1])
 except OSError:pass
 data['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 data['temperature_mC']=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
 return data
def model(kind,threads):
 if kind=='vad':
  c=s.VadModelConfig();c.silero_vad.model=str(ROOT/'models/silero_vad.onnx');c.silero_vad.min_silence_duration=.25;c.silero_vad.min_speech_duration=.25;c.sample_rate=16000;c.num_threads=threads
  return s.VoiceActivityDetector(c,buffer_size_in_seconds=60)
 if kind=='denoise':
  c=s.OnlineSpeechDenoiserConfig(model=s.OfflineSpeechDenoiserModelConfig(gtcrn=s.OfflineSpeechDenoiserGtcrnModelConfig(model=str(ROOT/'models/gtcrn_simple.onnx')),num_threads=threads,provider='cpu'))
  return s.OnlineSpeechDenoiser(c)
 if kind=='asr':
  return s.OfflineRecognizer.from_sense_voice(model=str(ASR/'model.int8.onnx'),tokens=str(ASR/'tokens.txt'),num_threads=threads,use_itn=True,provider='cpu')
 if kind=='speaker':
  c=s.SpeakerEmbeddingExtractorConfig(model=str(ROOT/'models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'),num_threads=threads,provider='cpu')
  assert c.validate();return s.SpeakerEmbeddingExtractor(c)
 raise ValueError(kind)
def infer(kind,m,x,save_audio=None):
 frames=[]
 if kind=='asr':
  st=m.create_stream();st.accept_waveform(16000,x);m.decode_stream(st)
  return {'text':st.result.text},frames
 if kind=='speaker':
  st=m.create_stream();st.accept_waveform(16000,x);st.input_finished();assert m.is_ready(st)
  vector=np.asarray(m.compute(st),np.float32);assert np.isfinite(vector).all()
  return {'embedding':vector.tolist(),'dimension':len(vector),'norm':float(np.linalg.norm(vector))},frames
 if kind=='vad':
  m.reset();segments=[];window=512
  for start in range(0,len(x),window):
   chunk=x[start:start+window];chunk=np.pad(chunk,(0,window-len(chunk)))
   t=time.perf_counter();m.accept_waveform(chunk);frames.append((time.perf_counter()-t)*1000)
   while not m.empty():
    seg=m.front;segments.append({'start_samples':seg.start,'samples':len(seg.samples)});m.pop()
  m.flush()
  while not m.empty():
   seg=m.front;segments.append({'start_samples':seg.start,'samples':len(seg.samples)});m.pop()
  return {'segments':segments,'speech_seconds':sum(z['samples'] for z in segments)/16000,'window_ms':32,'endpoint_silence_ms':250},frames
 if kind=='denoise':
  m.reset();out=[];window=m.frame_shift_in_samples
  for start in range(0,len(x),window):
   t=time.perf_counter();y=m(x[start:start+window],16000);frames.append((time.perf_counter()-t)*1000);out.append(np.asarray(y.samples,np.float32))
  out.append(np.asarray(m.flush().samples,np.float32));y=np.concatenate(out)
  assert len(y)>0 and np.isfinite(y).all()
  if save_audio:
   with wave.open(str(save_audio),'wb') as f:
    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((np.clip(y,-1,1)*32767).astype('<i2').tobytes())
  return {'output_samples':len(y),'input_samples':len(x),'output_rms':float(np.sqrt(np.mean(y*y))),'frame_shift_samples':window,'output_sha256':hashlib.sha256(y.tobytes()).hexdigest()},frames
def quantile(xs,q):return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['vad','denoise','asr','speaker'],required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=5);p.add_argument('--threads',type=int,default=1);p.add_argument('--seconds',type=float,default=0);p.add_argument('--ready');p.add_argument('--go');p.add_argument('--stop');p.add_argument('--paced',action='store_true');a=p.parse_args()
 samples=[]
 for name in ['zh','en','yue']:
  f=ROOT/'audio'/(name+'.wav')
  if f.exists():samples.append((name,read_wav(f)))
 assert samples
 # Deterministic controlled noise; never claim recorded home noise.
 rng=np.random.default_rng(20261018);clean=samples[0][1];noise=rng.standard_normal(len(clean)).astype(np.float32);noise*=np.sqrt(np.mean(clean**2)/np.mean(noise**2)) # 0dB SNR
 scale=max(1.,float(np.max(np.abs(clean+noise)))/.99);noisy=(clean+noise)/scale
 samples.extend([('zh-white-noise-0db',noisy),('silence',np.zeros(48000,np.float32))])
 if a.kind=='speaker':samples=[(n,x) for n,x in samples if n!='silence']
 t=time.perf_counter();ct=time.process_time();m=model(a.kind,a.threads);load=time.perf_counter()-t
 load_cpu=time.process_time()-ct;loaded=stats();runs=[]
 # Denoiser reset is version-dependent; validated in smoke before trials.
 if a.ready:Path(a.ready).write_text(json.dumps({'pid':os.getpid(),'kind':a.kind,'loaded':loaded}))
 if a.go:
  deadline=time.monotonic()+600
  while not Path(a.go).exists():
   if time.monotonic()>deadline:raise TimeoutError('go signal')
   time.sleep(.02)
 started=time.monotonic();iteration=0
 while True:
  n,x=samples[iteration%len(samples)];before=stats();t=time.perf_counter();ct=time.process_time();output,frames=infer(a.kind,m,x,Path(a.out).with_name(n+'-denoised.wav') if a.kind=='denoise' and iteration<len(samples) else None)
  elapsed=time.perf_counter()-t;cpu=time.process_time()-ct
  runs.append({'sample':n,'iteration':iteration,'seconds':elapsed,'cpu_seconds':cpu,'duration_seconds':len(x)/16000,'rtf':elapsed/(len(x)/16000),'start_monotonic':t,'end_monotonic':time.perf_counter(),'resources':stats(),'output':output,'frame_p50_ms':statistics.median(frames) if frames else None,'frame_p95_ms':quantile(frames,.95) if frames else None,'frame_max_ms':max(frames) if frames else None})
  iteration+=1
  if a.stop and Path(a.stop).exists():break
  if a.seconds:
   if time.monotonic()-started>=a.seconds:break
  elif iteration>=a.repeats*len(samples):break
  if a.paced:time.sleep(max(0,len(x)/16000-elapsed))
 hot=runs[len(samples):] or runs
 summary={'status':'complete','kind':a.kind,'runtime':s.__version__,'onnxruntime':s.onnxruntime_version() if callable(s.onnxruntime_version) else s.onnxruntime_version,'threads_requested':a.threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),'timing_scope':'offline replay, unflushed OS cache; first pass includes per-shape first-use; not end-to-end microphone latency','load_seconds':load,'load_cpu_seconds':load_cpu,'loaded_resources':loaded,'final_resources':stats(),'p50_seconds':statistics.median(r['seconds'] for r in hot),'p95_seconds':quantile([r['seconds'] for r in hot],.95),'rtf_p50':statistics.median(r['rtf'] for r in hot),'rtf_p95':quantile([r['rtf'] for r in hot],.95),'runs':runs}
 summary['by_sample']={n:{'calls':len(rs),'p50_seconds':statistics.median(r['seconds'] for r in rs),'p95_seconds':quantile([r['seconds'] for r in rs],.95),'rtf_p50':statistics.median(r['rtf'] for r in rs)} for n in sorted({r['sample'] for r in hot}) for rs in [[r for r in hot if r['sample']==n]]}
 summary['noise_generation']={'snr_db':0,'common_scale':scale,'clipping':False,'type':'synthetic white noise'}
 Path(a.out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='runs'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
