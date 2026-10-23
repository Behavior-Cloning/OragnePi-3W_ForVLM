import json,os,sys,time,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018')
import audio_worker as aw
sys.path.insert(0,'/opt/opi-vlm/asr-small-20261018')
import worker as sw
ROOT=Path('/opt/opi-vlm/asr-small-20261018');OUT=Path('/opt/opi-vlm/audio-streaming-20261018/paired');OUT.mkdir(parents=True,exist_ok=False)
manifest=json.loads((ROOT/'corpus/manifest.json').read_text());samples=manifest['samples'];indices=[0,2,8,9,16,20]
# Independent VAD proxy is precomputed before either ASR path; it is not a human true speech-end label.
vad=aw.model('vad',1,.25);proxy={}
for i in indices:
 row=samples[i];x=sw.readwav(ROOT/'corpus'/row['file']);seg,_=aw.infer('vad',vad,x)
 end=max((q['start_samples']+q['samples'] for q in seg['segments']),default=len(x))
 proxy[i]={'sample_end':end,'audio_seconds':end/16000,'speech_reference':'independent Silero VAD segment end; includes detector trailing silence; not human-annotated'}
para=sw.model('para',1);streamm=sw.model('stream',1);results=[]
def paced_play(samples_x,chunk=1280,cb=None):
 start=time.monotonic(); collected=[]
 for i in range(0,len(samples_x),chunk):
  x=samples_x[i:i+chunk];due=start+(i+len(x))/16000
  time.sleep(max(0,due-time.monotonic()));now=time.monotonic();collected.append(x.copy())
  if cb:cb(x,i,now,start)
 return start,np.concatenate(collected) if collected else np.zeros(0,np.float32)
for i in indices:
 row=samples[i];x=sw.readwav(ROOT/'corpus'/row['file']);end=proxy[i]['sample_end'];feed=np.concatenate([x,np.zeros(24000,np.float32)])
 # Batch endpoint mode: wait until the independent VAD proxy, then run the whole captured segment once.
 t0=time.monotonic();time.sleep(max(0,proxy[i]['audio_seconds']-(time.monotonic()-t0)))
 seg=para.create_stream();seg.accept_waveform(16000,feed[:end]);t=time.perf_counter();para.decode_stream(seg);batch_text=seg.result.text;batch_compute=time.perf_counter()-t;batch_proxy_elapsed=time.monotonic()-t0-proxy[i]['audio_seconds']
 results.append({'id':row['id'],'reference':row['text'],'vad_proxy_end_audio_seconds':proxy[i]['audio_seconds'],'vad_proxy_label':proxy[i]['speech_reference'],'batch':{'text':batch_text,'proxy_to_first_and_final_seconds':batch_proxy_elapsed+batch_compute,'inference_seconds':batch_compute,'final_event':'offline result returned after VAD proxy'}})
 # Streaming path: same WAV and 80 ms arrival chunks; retain every changed partial and model endpoint.
 st=streamm.create_stream();events=[];eps=[];last=None;clock={}
 def cb(chunk,offset,now,start):
  st.accept_waveform(16000,chunk)
  while streamm.is_ready(st):streamm.decode_stream(st)
  txt=streamm.get_result(st)
  if txt!=last:
   events.append({'monotonic':time.monotonic(),'audio_seconds':(offset+len(chunk))/16000,'text':txt})
  if streamm.is_endpoint(st):eps.append({'monotonic':time.monotonic(),'audio_seconds':(offset+len(chunk))/16000,'text':txt})
 start,_=paced_play(feed,cb=cb);proxy_mono=start+proxy[i]['audio_seconds'];before=streamm.get_result(st);tflush=time.monotonic();st.input_finished()
 while streamm.is_ready(st):streamm.decode_stream(st)
 final=streamm.get_result(st);final_mono=time.monotonic();events.append({'monotonic':final_mono,'audio_seconds':len(feed)/16000,'text':final,'event':'forced_flush'})
 first=next((e for e in events if e['text']),None)
 # Retrospective 320 ms unchanged text estimate; report separately from formal final event.
 stable=None
 for j,e in enumerate(events):
  if not e['text']:continue
  next_change=next((z for z in events[j+1:] if z['text']!=e['text']),None)
  if next_change is None and final_mono-e['monotonic']>=.32:stable=e;break
 results[-1]['streaming']={'text':final,'events':events,'first_partial_after_proxy_seconds':first['monotonic']-proxy_mono if first else None,'retrospective_stable_text_after_proxy_seconds':stable['monotonic']-proxy_mono if stable else None,'formal_final_after_proxy_seconds':final_mono-proxy_mono,'natural_endpoint_events':[{'after_proxy_seconds':e['monotonic']-proxy_mono,'audio_seconds':e['audio_seconds'],'text':e['text']} for e in eps],'forced_flush_changed_text':before!=final,'chunk_ms':80,'forced_flush':True}
# Save separate passes with a common reference; CER is informational and normalization-free (character Levenshtein).
def dist(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nd=[i]
  for j,y in enumerate(b,1):nd.append(min(d[j]+1,nd[-1]+1,d[j-1]+(x!=y)))
  d=nd
 return d[-1]
summary={'status':'complete','runtime':'sherpa-onnx 1.13.8','affinity':sorted(os.sched_getaffinity(0)),'threads':1,'samples':indices,'corpus':'public AISHELL-derived frozen corpus','time_reference':'independent Silero VAD segment end; includes detector trailing silence; not human speech-end labels','comparison':'Paraformer starts inference when captured audio reaches shared VAD proxy; streaming Zipformer receives same waveform at 80 ms cadence. One warm process/model per path, sequential replays, forced flush after appended 1.5s silence. The online path natural endpoint is reported separately.','rows':results}
for mode in ['batch','streaming']:
 rows=[r[mode] for r in results];refs=[r['reference'] for r in results];texts=[r['text'] for r in rows];ed=sum(dist(a,b) for a,b in zip(refs,texts));n=sum(len(s) for s in refs)
 summary[mode+'_accuracy']={'CER_percent':100*ed/n,'edit_count':ed,'reference_characters':n,'exact_sentences':sum(a==b for a,b in zip(refs,texts)),'sentences':len(refs)}
 if mode=='batch':summary[mode+'_latency']={'proxy_to_final_p50_seconds':statistics.median(r['proxy_to_first_and_final_seconds'] for r in rows),'per_sample_seconds':[r['proxy_to_first_and_final_seconds'] for r in rows]}
 else:
  for key,label in [('first_partial_after_proxy_seconds','proxy_to_first_partial'),('retrospective_stable_text_after_proxy_seconds','proxy_to_stable_text'),('formal_final_after_proxy_seconds','proxy_to_forced_final')]:
   v=[r[key] for r in rows if r[key] is not None];summary[label]={'p50_seconds':statistics.median(v) if v else None,'per_sample_seconds':v}
(OUT/'paired-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))
