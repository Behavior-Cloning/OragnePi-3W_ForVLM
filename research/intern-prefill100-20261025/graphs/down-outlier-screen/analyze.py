"""Offline, scenario-held-out channel screening; no NPU timing or quality claim."""
from pathlib import Path
import sys,json,hashlib,time,traceback,math
import numpy as np,torch
D=Path(__file__).resolve().parent;T=D.parents[1];S=T/'graphs/cases/c512-m32-l2';F=T/'down-i16-semantic'
sys.path.insert(0,str(T/'graphs/candidates-fp16'))
from graph_reference_conv import evaluate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(x):
 p=D/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
st={'status':'running','started':time.time(),'cases_done':0};save(st)
try:
 g=json.loads((S/'decoder.json').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();conv=json.loads((S/'conversion.json').read_text());cases=json.loads((F/'status.json').read_text())['cases'];assert len(cases)==48
 scenarios=sorted({c['scenario'] for c in cases},key=lambda s:hashlib.sha256(s.encode()).hexdigest());assert len(scenarios)==12
 train=set(scenarios[:6]);xs=[[],[]];metadata=[]
 with torch.inference_mode():
  for c in cases:
   p=F/c['id']/'input.f32';assert sha(p)==c['input_sha256'];raw=np.fromfile(p,dtype='<f4');off=0;feeds={}
   for key in conv['input_order']:
    n=g['Layers'][key];shape=n['parameters']['shape'];size=int(np.prod(shape));feeds[n['name']]=torch.from_numpy(raw[off:off+size].copy().reshape(shape));off+=size
   assert off==raw.size
   _,cache=evaluate(g,data,feeds,return_intermediates=True)
   for i in range(2):
    x=cache[f'b{i}_reshape_115'].clone();assert x.shape==(32,4864) and torch.isfinite(x).all();xs[i].append(x)
   metadata.append({**c,'split':'calibration' if c['scenario'] in train else 'heldout'})
   st['cases_done']+=1;save(st)
  result={'scope':'Offline FP32-source activations rounded to FP16; FP16-rounded weights, FP32 accumulation. Does not simulate NPU branch rounding, output quantization, captured NPU KV, system pos0, complete logits, latency or action quality.', 'source_sha256':{str(p.relative_to(T)):sha(p) for p in [Path(__file__),S/'decoder.json',S/'decoder.data',S/'conversion.json',F/'status.json',T/'graphs/candidates-fp16/graph_reference_conv.py']},'train_scenarios':sorted(train),'heldout_scenarios':sorted(set(scenarios)-train),'cases':metadata,'layers':[]}
  for i in range(2):
   x=torch.stack(xs[i]).half().float();w=torch.from_numpy(data[f'b{i}_fullconnect_116']['weight']).half().float();assert w.shape==(4864,896)
   cal=torch.tensor([m['split']=='calibration' for m in metadata]);score=x[cal].abs().amax((0,1));order=torch.argsort(score,descending=True,stable=True)
   fl=math.floor(math.log2(32767/float(w.abs().max())));wq=(w*2**fl).round().clamp(-32768,32767)/2**fl
   ref=x@w;rows=[]
   for rng in [8,16]:
    af=12 if rng==8 else 11;xq=(x*2**af).round().clamp(-32768,32767)/2**af;base=xq@wq
    for k in [0,1,4,8,16,32,64,128]:
     ix=order[:k];pred=base.clone()
     if k:pred+=x[:,:,ix]@w[ix]-xq[:,:,ix]@wq[ix]
     assert torch.isfinite(pred).all()
     diff=pred-ref;rel=torch.linalg.vector_norm(diff,dim=(1,2))/torch.linalg.vector_norm(ref,dim=(1,2)).clamp_min(1e-30)
     sat=((x*2**af).round()>32767)|((x*2**af).round()<-32768);sat[:,:,ix]=False
     rows.append({'activation_range':rng,'weight_fl':fl,'fp16_channels':k,'channel_ids':ix.tolist(),'case_metrics':[{'id':m['id'],'split':m['split'],'relative_l2':float(rel[j]),'max_abs_error':float(diff[j].abs().max()),'unprotected_clips':int(sat[j].sum())} for j,m in enumerate(metadata)]})
   layer={'layer':i+2,'shape':list(x.shape),'maxabs':float(x.abs().max()),'calibration_maxabs':float(x[cal].abs().max()),'heldout_maxabs':float(x[~cal].abs().max()),'channels_exceeding':{str(v):int((x.abs().amax((0,1))>v).sum()) for v in [8,16,32,128,512]},'top_channels':[{'channel':int(c),'calibration_maxabs':float(score[c]),'heldout_maxabs':float(x[~cal,:,c].abs().max())} for c in order[:32]],'variants':rows};result['layers'].append(layer)
  (D/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  st.update(status='complete',ended=time.time(),analysis_sha256=sha(D/'analysis.json'));save(st)
except BaseException:st.update(status='failed',error=traceback.format_exc(),ended=time.time());save(st);raise
