import json,os,sys,time,hashlib,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018');import audio_worker as aw
sys.path.insert(0,'/opt/opi-vlm/asr-small-20261018');import worker as sw
R=Path('/opt/opi-vlm/asr-small-20261018');OUT=Path('/opt/opi-vlm/audio-streaming-20261018/denoise');OUT.mkdir(parents=True,exist_ok=False);man=json.loads((R/'corpus/manifest.json').read_text());ids=[24,25,26,27,28,29,30,31]
os.sched_setaffinity(0,{6,7});d=aw.model('denoise',1);m=sw.model('para',1);rows=[]
for i in ids:
 q=man['samples'][i];x=sw.readwav(R/'corpus'/q['file']);
 t=time.perf_counter();d.reset();ys=[]
 for p in range(0,len(x),256):ys.append(np.asarray(d(x[p:p+256],16000).samples,np.float32))
 ys.append(np.asarray(d.flush().samples,np.float32));y=np.concatenate(ys);dt=time.perf_counter()-t
 results={}
 for name,z in [('noisy',x),('denoised',y)]:
  st=m.create_stream();st.accept_waveform(16000,z);t=time.perf_counter();m.decode_stream(st);results[name]={'text':st.result.text,'seconds':time.perf_counter()-t}
 rows.append({'id':q['id'],'reference':q['text'],'duration':len(x)/16000,'denoise_seconds':dt,'denoise_rtf':dt/(len(x)/16000),'denoised_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'results':results})
def dist(a,b):
 d=list(range(len(b)+1))
 for x in a:
  n=[d[0]+1]
  for j,y in enumerate(b,1):n.append(min(d[j]+1,n[-1]+1,d[j-1]+(x!=y)))
  d=n
 return d[-1]
summary={'status':'complete','affinity':[6,7],'threads':1,'scope':'4 public AISHELL sentences x synthetic 10dB/0dB white noise; GTCRN then Paraformer; no human/home noise','rows':rows}
for label,ids0 in [('10db',ids[::2]),('0db',ids[1::2])]:
 rs=[rows[ids.index(i)] for i in ids0]
 for method in ['noisy','denoised']:
  e=sum(dist(r['reference'],r['results'][method]['text']) for r in rs);n=sum(len(r['reference']) for r in rs)
  summary.setdefault('accuracy',{}).setdefault(label,{})[method]={'CER_percent':100*e/n,'edits':e,'reference_characters':n,'exact_sentences':sum(r['reference']==r['results'][method]['text'] for r in rs)}
(OUT/'asr-compare.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary['accuracy'],ensure_ascii=False))
