"""Small public-demo diagnostic: no household identity/recognition acceptance claim."""
import hashlib,json,statistics,time,wave
from pathlib import Path
import numpy as np
import audio_worker as w
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality-v1';OUT.mkdir(exist_ok=False)
def save_wav(name,x):
 with wave.open(str(OUT/name),'wb') as f:
  f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((np.clip(x,-1,1)*32767).astype('<i2').tobytes())
def norm(x):return x/np.linalg.norm(x)
def sisdr(ref,est):
 # Alignment search only for offline quality; excluded from inference timing.
 best=(-1.,0)
 for lag in range(-512,513):
  r=ref[max(0,-lag):min(len(ref),len(est)-lag):8];e=est[max(0,lag):min(len(est),len(ref)+lag):8]
  if len(r)!=len(e) or not len(r):continue
  score=float(np.dot(r,e)/(np.linalg.norm(r)*np.linalg.norm(e)+1e-12))
  if score>best[0]:best=(score,lag)
 lag=best[1];r=ref[max(0,-lag):min(len(ref),len(est)-lag)];e=est[max(0,lag):min(len(est),len(ref)+lag)];r=r-r.mean();e=e-e.mean();target=np.dot(e,r)/(np.dot(r,r)+1e-12)*r
 return {'si_sdr_db':float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((e-target)**2)+1e-12))),'alignment_lag_samples':lag}
def main():
 result={'scope':'official public demo smoke, no human reference transcripts or household evaluation','speaker':{},'denoise':{}}
 m=w.model('speaker',1);vectors={};timings={}
 for f in sorted((ROOT/'speaker-samples').glob('*.wav')):
  x=w.read_wav(f);t=time.perf_counter();v,_=w.infer('speaker',m,x);timings[f.name]=time.perf_counter()-t;vectors[f.name]=norm(np.asarray(v['embedding'],np.float32))
 ids=['fangjun','leijun'];centers={i:norm(np.mean([v for n,v in vectors.items() if n.startswith(i+'-sr-')],axis=0)) for i in ids}
 tests=[]
 for name,v in vectors.items():
  if '-test-' not in name:continue
  scores=sorted(((i,float(np.dot(v,c))) for i,c in centers.items()),key=lambda a:a[1],reverse=True)
  predicted=scores[0][0] if scores[0][1]>=.6 and scores[0][1]-scores[1][1]>=.1 else 'unknown'
  label=name.split('-')[0];expected=label if label in ids else 'unknown'
  tests.append({'file':name,'expected':expected,'predicted':predicted,'scores':dict(scores),'seconds':timings[name],'correct':expected==predicted})
 result['speaker']={'registered':ids,'held_out_identity':'liudehua','threshold':.6,'margin':.1,'threshold_source':'predeclared illustrative policy, not calibrated','enrollment_disjoint_from_test':True,'tests':tests,'correct':sum(t['correct'] for t in tests),'total':len(tests),'resources':w.stats()}
 del m
 clean=w.read_wav(ROOT/'audio/zh.wav');rng=np.random.default_rng(20261018);noise=rng.standard_normal(len(clean)).astype(np.float32);noise*=np.sqrt(np.mean(clean**2)/np.mean(noise**2));scale=max(1.,float(np.max(np.abs(clean+noise)))/.99);ref=clean/scale;noisy=(clean+noise)/scale
 save_wav('clean-scaled.wav',ref);save_wav('noisy-0db.wav',noisy)
 d=w.model('denoise',1);t=time.perf_counter();info,_=w.infer('denoise',d,noisy,OUT/'denoised.wav');elapsed=time.perf_counter()-t;enhanced=w.read_wav(OUT/'denoised.wav')
 result['denoise']={'seconds':elapsed,'input_seconds':len(noisy)/16000,'output':info,'synthetic_noise':'0dB white noise, common scaling, no clipping','before':sisdr(ref,noisy),'after':sisdr(ref,enhanced)}
 del d
 asr=w.model('asr',1);texts={}
 for n,x in [('clean',ref),('noisy',noisy),('denoised',enhanced)]:
  t=time.perf_counter();r,_=w.infer('asr',asr,x);texts[n]={'text':r['text'],'seconds':time.perf_counter()-t}
 result['denoise']['asr_comparison']=texts
 result['denoise']['quality_limit']='Transcript agreement is a diagnostic; clean-model output is not ground truth. SI-SDR uses synthetic clean reference; no subjective/MOS/PESQ claim.'
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
