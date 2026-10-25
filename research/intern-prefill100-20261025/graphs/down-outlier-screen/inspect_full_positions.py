"""Inspect all positions from source-Q8 FP32 full-sequence paths, no NPU simulation."""
from pathlib import Path
import json,hashlib,sys,time,traceback,types
import numpy as np,torch,gguf
import torch.nn.functional as F
D=Path(__file__).resolve().parent;T=D.parents[1];R=T.parents[1];sys.path.insert(0,str(R/'npu/kv-cache'))
from trained_probe import TrainedDecoder
M=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf');P=T/'fullconv-quality/semantic-inputs.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
st={'status':'running','started':time.time(),'completed':0}
def save():
 q=D/'full-position-status.tmp';q.write_text(json.dumps(st,indent=2)+'\n');q.replace(D/'full-position-status.json')
save()
try:
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert sha(M)=='e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f'
 reader=gguf.GGUFReader(M);embt=next(t for t in reader.tensors if t.name=='token_embd.weight');emb=gguf.dequantize(embt.data,embt.tensor_type);layers=[TrainedDecoder(reader,i,True).eval() for i in range(4)];captured={}
 for i in [2,3]:
  original=layers[i].finish
  def finish(self,hidden,attention,idx=i,orig=original):
   h=hidden+F.linear(attention.transpose(1,2).reshape(1,-1,self.dim),self.o);z=self.norm(h,self.norm2);x=F.silu(F.linear(z,self.gate))*F.linear(z,self.up);captured[idx]=x[0].clone();out=orig(hidden,attention);torch.testing.assert_close(out,h+F.linear(x,self.down),rtol=0,atol=0);return out
  layers[i].finish=types.MethodType(finish,layers[i])
 fixtures=json.loads(P.read_text())['quality'];split=json.loads((D/'analysis.json').read_text());train=set(split['train_scenarios']);rows=[];vals={2:[],3:[]}
 with torch.inference_mode():
  for q in fixtures:
   ids=q['tokens'];assert len(ids)==q['static_tokens']+q['body_tokens']+8;assert hashlib.sha256(np.array(ids,dtype='<u4').tobytes()).hexdigest()==q['tokens_sha256'];h=torch.from_numpy(emb[ids][None].copy())
   for layer in layers:h=layer.full(h)
   rec={'id':q['id'],'scenario':q['case'],'split':'calibration' if q['case'] in train else 'heldout','static_tokens':q['static_tokens'],'body_tokens':q['body_tokens'],'template_dynamic_tokens':8,'total_tokens':len(ids),'layers':{}}
   for i in [2,3]:
    x=captured[i];assert torch.isfinite(x).all();vals[i].append(x);parts={}
    for name,a,b in [('prefix',0,q['static_tokens']),('dynamic',q['static_tokens'],len(ids))]:
     y=x[a:b];parts[name]={'maxabs':float(y.abs().max()),'channels_gt8':torch.where(y.abs().amax(0)>8)[0].tolist(),'channels_gt16':torch.where(y.abs().amax(0)>16)[0].tolist(),'elements_gt16':int((y.abs()>16).sum())}
    rec['layers'][str(i)]=parts
   rows.append(rec);st['completed']+=1;save()
  summaries=[]
  for i in [2,3]:
   cal=torch.cat([x for x,r in zip(vals[i],rows) if r['split']=='calibration']);score=cal.abs().amax(0);order=torch.argsort(score,descending=True,stable=True);rank=torch.empty_like(order);rank[order]=torch.arange(4864);issues=[]
   for x,r in zip(vals[i],rows):
    for name,a,b in [('prefix',0,r['static_tokens']),('dynamic',r['static_tokens'],r['total_tokens'])]:
     y=x[a:b];counts=[]
     for rng in [8,16]:
      large=y.abs()>rng;channels=torch.where(large.any(0))[0]
      counts.append({'range':rng,'max_rank_1based':int(rank[channels].max()+1) if len(channels) else 0,'channels':channels.tolist(),'unprotected_by_k':{str(k):int(large[:,order[k:]].sum()) for k in [1,4,8,16,32,64,128]}})
     issues.append({'id':r['id'],'split':r['split'],'part':name,'counts':counts})
   summaries.append({'layer':i,'top32':[{'channel':int(c),'calibration_maxabs':float(score[c])} for c in order[:32]],'coverage':issues})
  result={'scope':'Independent full-sequence source-Q8 FP32 model through layer3. No exported-quantized/NPU history, branch rounding, output quantization or latency. Includes every input position; source differs from imported-graph FP32 screen and should not be pooled numerically. Shared fixed system appears in both splits; no claim of unseen system generalization.','sources':{'script':sha(Path(__file__)),'fixture':sha(P),'model':sha(M),'trained_probe':sha(R/'npu/kv-cache/trained_probe.py'),'split_source':sha(D/'analysis.json')},'rows':rows,'layers':summaries}
  (D/'full-position-analysis.json').write_text(json.dumps(result,indent=2)+'\n');st.update(status='complete',ended=time.time(),result_sha256=sha(D/'full-position-analysis.json'));save()
except BaseException:st.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
