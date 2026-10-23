from pathlib import Path
import json,hashlib,math,inspect
import torch,numpy as np,onnx,onnxruntime as ort
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer,Gemma2RotaryEmbedding
r=Path(__file__).resolve().parent;src=r.parent/'vlm-molmo-paligemma-20261023/paligemma/model';out=Path('/opt/opi-lab/paligemma2-npu-compat-20261023');out.mkdir(exist_ok=True)
config=Gemma2Config(**json.loads((src/'config.json').read_text())['text_config']);config._attn_implementation='eager'
index=json.loads((src/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.0.'
state={};hashes={}
for file in set(v for k,v in index.items() if k.startswith(prefix)):
 with safe_open(src/file,framework='pt',device='cpu') as f:
  for name in index:
   if name.startswith(prefix) and index[name]==file:
    bf=f.get_tensor(name);hashes[name]=hashlib.sha256(bf.view(torch.uint16).numpy().tobytes()).hexdigest();state[name[len(prefix):]]=bf.float()
torch.set_num_threads(2)
ref=Gemma2DecoderLayer(config,0).eval();ref.load_state_dict(state,strict=True,assign=True)
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.ranges={}
  mapping={'q':'self_attn.q_proj.weight','k':'self_attn.k_proj.weight','v':'self_attn.v_proj.weight','o':'self_attn.o_proj.weight','gate':'mlp.gate_proj.weight','up':'mlp.up_proj.weight','down':'mlp.down_proj.weight','n0':'input_layernorm.weight','n1':'post_attention_layernorm.weight','n2':'pre_feedforward_layernorm.weight','n3':'post_feedforward_layernorm.weight'}
  for n,k in mapping.items():self.register_buffer(n,state[k])
 def norm(self,x,w):
  # Algebraically same norm; avoid underflow of tiny eps in FP16 deployment.
  d=x.abs().amax(-1,keepdim=True).clamp_min(config.rms_norm_eps**.5);z=x/d;eps=(config.rms_norm_eps**.5/d).square()
  return z*torch.rsqrt((z*z).mean(-1,keepdim=True)+eps)*(1+w)
 def lin(self,x,n):
  z=F.linear(x,getattr(self,n))
  if not torch.onnx.is_in_onnx_export():
   old=self.ranges.get(n,[0,0]);self.ranges[n]=[max(old[0],float(x.abs().max())),max(old[1],float(z.abs().max()))]
  return z
 def project(self,x,cos,sin):
  x=self.norm(x,self.n0);q=self.lin(x,'q').reshape(1,-1,8,256).transpose(1,2);k=self.lin(x,'k').reshape(1,-1,4,256).transpose(1,2);v=self.lin(x,'v').reshape(1,-1,4,256).transpose(1,2)
  rot=lambda z:z*cos+torch.cat((-z[...,128:],z[...,:128]),-1)*sin
  return rot(q),rot(k),v
 def finish(self,x,a):
  x=x+self.norm(self.lin(a.transpose(1,2).reshape(1,-1,2048),'o'),self.n1)
  z=self.norm(x,self.n2);z=self.lin(F.gelu(self.lin(z,'gate'),approximate='tanh')*self.lin(z,'up'),'down')
  return x+self.norm(z,self.n3)
 def attn(self,q,k,v,mask):
  k=k.repeat_interleave(2,1);v=v.repeat_interleave(2,1)
  scores=50*torch.tanh((q@k.transpose(-1,-2))*(config.query_pre_attn_scalar**-.5)/50)
  return torch.softmax(scores+mask,dim=-1)@v
 def forward(self,hidden,past_k,past_v,mask,rope_cos,rope_sin):
  q,k,v=self.project(hidden,rope_cos,rope_sin)
  return self.finish(hidden,self.attn(q,torch.cat([past_k,k],2),torch.cat([past_v,v],2),mask)),k,v
 def full(self,x,cos,sin,mask):
  q,k,v=self.project(x,cos,sin);return self.finish(x,self.attn(q,k,v,mask))
block=Block().eval();tok=Tokenizer.from_file(str(src/'tokenizer.json'));rotary=Gemma2RotaryEmbedding(config)
texts=['你好，请介绍一下自己。','我今天有点难过，陪我聊两句。','桌子上有什么东西？']
fixtures=[];checks=[];cap=32
for case,text in enumerate(texts):
 ids=[2]+tok.encode(text,add_special_tokens=False).ids
 key='language_model.model.embed_tokens.weight'
 with safe_open(src/index[key],framework='pt',device='cpu') as sf:
  emb=sf.get_slice(key);x=torch.stack([emb[i:i+1][0] for i in ids]).float()[None]*math.sqrt(config.hidden_size)
 posids=torch.arange(len(ids))[None];cos,sin=rotary(x,posids);cos4=cos[:,None];sin4=sin[:,None]
 for mode in ['causal','prefix_bidirectional']:
  n=len(ids);mask=torch.zeros(1,1,n,n) if mode=='prefix_bidirectional' else torch.triu(torch.full((1,1,n,n),-10000.),diagonal=1)
  with torch.inference_mode():
   oracle=ref(x,position_embeddings=(cos,sin),attention_mask=mask)[0];own=block.full(x,cos4,sin4,mask)
  err=float((own-oracle).abs().max());rel=float(torch.linalg.vector_norm(own-oracle)/torch.linalg.vector_norm(oracle));assert rel<3e-6,(case,mode,rel)
  checks.append(dict(case=case,mode=mode,tokens=n,max_abs=err,relative_l2=rel))
  with torch.inference_mode():q,k,v=block.project(x,cos4,sin4)
  for pos in sorted(set([0,1,n-1])):
   inds=list(range(n)) if mode=='prefix_bidirectional' else list(range(pos));inds=[j for j in inds if j!=pos]
   pk=torch.zeros(1,4,cap,256);pv=torch.zeros_like(pk);pk[:,:,:len(inds)]=k[:,:,inds];pv[:,:,:len(inds)]=v[:,:,inds];m=torch.full((1,1,1,cap+1),-10000.);m[...,:len(inds)]=0;m[...,-1]=0
   inputs=[x[:,pos:pos+1],pk,pv,m,cos4[:,:,pos:pos+1],sin4[:,:,pos:pos+1]]
   with torch.inference_mode():z=block(*inputs)[0]
   np.testing.assert_allclose(z.numpy(),oracle[:,pos:pos+1].numpy(),atol=1e-4,rtol=3e-4)
   fixtures.append(dict(case=case,mode=mode,pos=pos,inputs=inputs,reference=oracle[:,pos:pos+1].numpy()))
# Sliding boundary, independent of short calibration input lengths.
positions=np.array([0,1,4095,4096,4097]);current=4097
allowed=(positions<=current)&(positions>current-config.sliding_window)
assert allowed.tolist()==[False,False,True,True,True]
inputs=['hidden','past_k','past_v','mask','rope_cos','rope_sin'];outputs=['hidden_out','new_k','new_v'];path=out/'layer0.onnx'
torch.onnx.export(block,tuple(fixtures[0]['inputs']),str(path),input_names=inputs,output_names=outputs,opset_version=13,dynamo=False)
m=onnx.load(path);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,path)
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
entries=[];meta=[]
for i,case in enumerate(fixtures):
 feed={n:v.numpy() for n,v in zip(inputs,case['inputs'])};z=session.run(outputs,feed)[0];reference=case['reference'];np.testing.assert_allclose(z,reference,atol=1e-4,rtol=3e-4)
 names=[]
 for name,value in feed.items():p=f's{i}_{name}.npy';np.save(out/p,value);names.append(p)
 np.save(out/f's{i}_reference.npy',reference);entries.append(' '.join(names));meta.append({k:v for k,v in case.items() if k not in ['inputs','reference']})
 meta[-1].update(onnx_max_abs=float(np.abs(z-reference).max()),onnx_relative_l2=float(np.linalg.norm(z-reference)/np.linalg.norm(reference)))
(out/'layer0-dataset.txt').write_text('\n'.join(entries)+'\n');(out/'fixtures.json').write_text(json.dumps(meta,indent=2))
weight_hashes={n:hashlib.sha256(getattr(block,n).numpy().T.copy().tobytes()).hexdigest() for n in block.ranges}
summary=dict(source='official BF16 safetensors, only actual layer0 loaded',weight_bf16_sha256=hashes,architecture=dict(dim=config.hidden_size,heads=config.num_attention_heads,kv=config.num_key_value_heads,head_dim=config.head_dim,eps=config.rms_norm_eps,softcap=config.attn_logit_softcapping,query_scalar=config.query_pre_attn_scalar,rope_theta=config.rope_theta,layer0_type=config.layer_types[0],sliding_window=config.sliding_window),torch_official_checks=checks,onnx_checks=meta,calibration_ranges=block.ranges,import_weight_sha256=weight_hashes,onnx_bytes=path.stat().st_size,onnx_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),sliding_mask_check=dict(positions=positions.tolist(),current=current,allowed=allowed.tolist()),scope='text layer0 only; no image encoder/projector; all-prefix text checks are not image fixtures; long-window mask checked as logic only, not whole4096-token layer')
(out/'export-summary.json').write_text(json.dumps(summary,indent=2));(r/'export-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
