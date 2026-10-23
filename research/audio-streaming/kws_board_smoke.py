import json,os,sys,time,wave,hashlib,statistics
from pathlib import Path
import numpy as np
import sherpa_onnx as s
D=Path('/opt/opi-vlm/audio-streaming-20261018/kws')
M=D/'model'
kws=s.KeywordSpotter(tokens=str(M/'tokens.txt'),encoder=str(M/'encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx'),decoder=str(M/'decoder-epoch-13-avg-2-chunk-16-left-64.onnx'),joiner=str(M/'joiner-epoch-13-avg-2-chunk-16-left-64.onnx'),keywords_file=str(D/'keywords.txt'),num_threads=1,keywords_threshold=.25)
def read(p):
 with wave.open(str(p),'rb') as f:
  assert f.getframerate()==16000 and f.getnchannels()==1
  return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
rows=[]; root=M/'test_wavs'
for fn,expected in [('zh_0.wav','周望军'),('zh_1.wav','女儿'),('zh_2.wav','法国'),('zh_3.wav','见面会'),('zh_4.wav','落实'),('zh_5.wav','none'),('zh_6.wav','none'),('en_0.wav','none')]:
 x=read(root/fn); st=kws.create_stream(); events=[]; t=time.perf_counter(); cpu=time.process_time()
 for i in range(0,len(x),1280):
  st.accept_waveform(16000,x[i:i+1280])
  while kws.is_ready(st):kws.decode_stream(st)
  r=kws.get_result(st)
  if r:events.append({'at_seconds':time.perf_counter()-t,'raw':r})
 rows.append({'file':fn,'expected':expected,'duration':len(x)/16000,'events':events,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu})
# 3 s silence negative
st=kws.create_stream();x=np.zeros(48000,np.float32);t=time.perf_counter()
for i in range(0,len(x),1280):
 st.accept_waveform(16000,x[i:i+1280])
 while kws.is_ready(st):kws.decode_stream(st)
 r=kws.get_result(st)
print(json.dumps({'runtime':s.__version__,'model':'sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20','quant':'int8 encoder, fp32 decoder/joiner','threads':1,'affinity':sorted(os.sched_getaffinity(0)),'threshold':.25,'rows':rows,'resources':open('/proc/self/status').read().split('VmRSS:')[1].splitlines()[0].strip()},ensure_ascii=False,indent=2))
