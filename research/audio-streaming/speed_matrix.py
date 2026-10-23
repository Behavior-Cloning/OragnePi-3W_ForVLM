import json,os,sys,time,statistics,hashlib,wave,resource
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018');import audio_worker as aw
sys.path.insert(0,'/opt/opi-vlm/asr-small-20261018');import worker as sw
CORP=Path('/opt/opi-vlm/asr-small-20261018/corpus');MAN=json.loads((CORP/'manifest.json').read_text());OUT=Path('/opt/opi-vlm/audio-streaming-20261018/speed');OUT.mkdir(parents=True,exist_ok=False)
configs=[('A55',[2,3],1),('A55',[2,3],2),('A76',[6,7],1),('A76',[6,7],2)]
# Duplicate a short utterance for deterministic primitive runtime comparison.
rows=[]
for name,cpus,nthreads in configs:
 os.sched_setaffinity(0,set(cpus))
 for kind in ['vad','denoise','para']:
  start=time.perf_counter();cpu0=time.process_time()
  if kind in ['vad','denoise']:m=aw.model('vad' if kind=='vad' else 'denoise',nthreads,.5)
  else:m=sw.model('para',nthreads)
  load=time.perf_counter()-start
  # pre-warm model on the exact first utterance, exclude warm-up from latency but retain it in log.
  ids=[0,2,8,9,16,20];runrows=[]
  for ix in ids:
   r=MAN['samples'][ix];x=sw.readwav(CORP/r['file']);start=time.perf_counter();ct=time.process_time()
   if kind=='para':
    st=m.create_stream();st.accept_waveform(16000,x);m.decode_stream(st);out={'text':st.result.text}
   else:out,frames=aw.infer(kind,m,x)
   elapsed=time.perf_counter()-start;cpu=time.process_time()-ct
   runrows.append({'id':r['id'],'seconds':elapsed,'cpu_seconds':cpu,'audio_seconds':len(x)/16000,'rtf':elapsed/(len(x)/16000),'text':out.get('text'),'reference':r['text'],'frame_p95_ms':aw.quantile(frames,.95) if kind!='para' else None,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'temp_mC':aw.stats()['temperature_mC']})
  rows.append({'cluster':name,'affinity':cpus,'threads':nthreads,'kind':kind,'load_seconds':load,'load_cpu_seconds':time.process_time()-cpu0,'p50_seconds':statistics.median(q['seconds'] for q in runrows),'p95_seconds':aw.quantile([q['seconds'] for q in runrows],.95),'rtf_p50':statistics.median(q['rtf'] for q in runrows),'max_rss_kib':max(q['rss_kib'] for q in runrows),'max_temp_mC':max(q['temp_mC'] for q in runrows),'rows':runrows})
  del m
# Paired CER for Para configurations: common exact utterances and normalization-free CER.
def dist(a,b):
 d=list(range(len(b)+1))
 for x in a:
  nd=[nd0:=d[0]+1]
  for j,y in enumerate(b,1):nd.append(min(d[j]+1,nd[-1]+1,d[j-1]+(x!=y)))
  d=nd
 return d[-1]
for q in rows:
 if q['kind']=='para':
  e=sum(dist(z['reference'],z['text']) for z in q['rows']);n=sum(len(z['reference']) for z in q['rows']);q['CER_percent']=100*e/n;q['exact_sentences']=sum(z['reference']==z['text'] for z in q['rows'])
summary={'status':'complete','scope':'six fixed clean public Chinese utterances; one warm model load per cluster/thread configuration; no mic/end-to-end; VAD and denoise fixed offline probe; temperature and core frequencies fluctuate naturally','runtime':'sherpa-onnx 1.13.8','configs':rows}
(OUT/'matrix.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in rows],ensure_ascii=False))
