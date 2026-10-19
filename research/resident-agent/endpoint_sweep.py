"""Offline endpoint calibration on the same denoised synthetic fixture.
No wall-clock latency claim: compare split counts and delivered sample indices.
"""
import json,sys,time
from pathlib import Path
import agent_replay as a
import numpy as np
root=Path(__file__).resolve().parent
out=root/sys.argv[1];out.mkdir(exist_ok=False)
a.os.sched_setaffinity(0,{6})
dn=a.aw.model('denoise',1);thresholds=[.25,.5,.8,1.,1.2,1.4,1.6]
rows=[];texts=json.loads((root/'speech-texts.json').read_text())
for sid in range(1,13):
 x=a.wav(root/'speech'/('%02d.wav'%sid));y=np.concatenate([x,np.zeros(3*16000,np.float32)]);clean=[];dn.reset();t=time.monotonic()
 for offset in range(0,len(y),256):
  block=y[offset:offset+256]
  if len(block)<256:block=np.pad(block,(0,256-len(block)))
  clean.append(np.asarray(dn(block,16000).samples,np.float32))
 clean=np.concatenate(clean);denoise_compute=time.monotonic()-t
 for threshold in thresholds:
  vad=a.aw.model('vad',1,threshold);segments=[]
  for offset in range(0,len(clean)-511,512):
   vad.accept_waveform(clean[offset:offset+512])
   while not vad.empty():
    seg=vad.front;end=seg.start+len(seg.samples);segments.append({'start':seg.start,'end':end,'delivered':offset+512,'endpoint_wait_s':(offset+512-end)/16000,'file_end_to_endpoint_s':(offset+512-len(x))/16000});vad.pop()
  rows.append({'source_id':sid,'reference':texts[sid],'threshold_s':threshold,'segments':segments,'denoise_compute_s':denoise_compute,'file_duration_s':len(x)/16000})
summary={str(th):{'one_segment_count':sum(len(x['segments'])==1 for x in rows if x['threshold_s']==th),'sample_count':12,'failed_source_ids':[x['source_id'] for x in rows if x['threshold_s']==th and len(x['segments'])!=1]} for th in thresholds}
passing=[th for th in thresholds if summary[str(th)]['one_segment_count']==12]
result={'scope':'offline sample-index sweep; synthetic fixture only; not wall-clock end-to-end latency or generalization test','summary':summary,'recommended_threshold_s':min(passing) if passing else None,'rows':rows}
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(summary),flush=True)
