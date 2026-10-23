import json,os,time,wave,hashlib
from pathlib import Path
import numpy as np
import sherpa_onnx as s
D=Path('/opt/opi-vlm/audio-streaming-20261018/kws');M=D/'model';C=D/'candidate';
kws=s.KeywordSpotter(tokens=str(M/'tokens.txt'),encoder=str(M/'encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx'),decoder=str(M/'decoder-epoch-13-avg-2-chunk-16-left-64.onnx'),joiner=str(M/'joiner-epoch-13-avg-2-chunk-16-left-64.onnx'),keywords_file=str(C/'keywords.txt'),num_threads=1,keywords_threshold=.25)
def read(p):
 with wave.open(str(p),'rb') as f:
  assert f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2
  return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
rows=[]
for group in ['tts','corpus']:
 for path in sorted((C/group).glob('*.wav')):
  x=read(path);st=kws.create_stream();events=[];start=time.monotonic();cpu=time.process_time()
  for i in range(0,len(x),1280):
   st.accept_waveform(16000,x[i:i+1280])
   while kws.is_ready(st):kws.decode_stream(st)
   r=kws.get_result(st)
   if r:events.append({'at_wallclock_seconds':time.monotonic()-start,'audio_seconds':min(i+1280,len(x))/16000,'output':r})
  rows.append({'group':group,'file':path.name,'duration_seconds':len(x)/16000,'events':events,'wallclock_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'rtf':(time.monotonic()-start)/(len(x)/16000)})
print(json.dumps({'model':'sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20','keyword':'小白小白','phoneme_sequence':'x iǎo b ái x iǎo b ái','threshold':.25,'threads':1,'affinity':sorted(os.sched_getaffinity(0)),'scope':'synthetic Microsoft Huihui Desktop positives/negative phrases and public AISHELL speech negatives; replay is offline rapid feed, not microphone timing','rows':rows},ensure_ascii=False,indent=2))
