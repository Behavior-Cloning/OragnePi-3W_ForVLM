import torch,json,math,numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,torch_recurrent_gated_delta_rule
ROOT=Path('/opt/opi-lab/minicpm-v46-npu-20261023');MODEL=Path('/mnt/f/Files/OPi-Lab/models/MiniCPM-V-4.6')
CFG=Qwen3_5TextConfig(**json.loads((MODEL/'config.json').read_text())['text_config']);CFG._attn_implementation='eager'
def stable_norm(x,w,centered=True):
 scale=x.abs().amax(-1,keepdim=True).clamp(min=.001)
 y=x/scale
 y=y*torch.rsqrt((y*y).mean(-1,keepdim=True)+(1e-6/scale)/scale)
 return y*(1+w if centered else w)

class LinearLayer(torch.nn.Module):
 def __init__(self,layer):super().__init__();self.layer=layer
 def forward(self,hidden,conv,state):
  l=self.layer;a=l.linear_attn;x=stable_norm(hidden,l.input_layernorm.weight)
  qkv=a.in_proj_qkv(x).transpose(1,2)
  new_conv=torch.cat([conv[:,:,1:],qkv],dim=-1)
  mixed=torch.nn.functional.silu(sum(new_conv[:,:,j]*a.conv1d.weight[:,0,j] for j in range(4)))
  q,k,v=[t.reshape(1,16,1,128) for t in mixed.split(2048,dim=-1)]
  q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*(128**-0.5)
  k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
  g=-a.A_log.exp().reshape(1,16,1,1)*torch.nn.functional.softplus(a.in_proj_a(x)+a.dt_bias).reshape(1,16,1,1)
  beta=a.in_proj_b(x).sigmoid().reshape(1,16,1,1)
  decayed=state*torch.exp(g);delta=(v-k@decayed)*beta;new_state=decayed+k.transpose(-1,-2)@delta
  out=(q@new_state).reshape(16,128);z=a.in_proj_z(x).reshape(16,128)
  out=(stable_norm(out,a.norm.weight,False)*torch.nn.functional.silu(z)).reshape(1,1,2048);h=hidden+a.out_proj(out)
  return h+l.mlp(stable_norm(h,l.post_attention_layernorm.weight)),new_conv,new_state

def load_layer(i):
 l=Qwen3_5DecoderLayer(CFG,i).eval();prefix='model.language_model.layers.'+str(i)+'.'
 with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as f:
  state={k[len(prefix):]:f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}
 l.load_state_dict(state,strict=True);return l
if __name__=='__main__':
 from types import SimpleNamespace
 torch.set_num_threads(4);torch.manual_seed(46);l=load_layer(0);m=LinearLayer(l).eval();p=ROOT/'linear0-stable';p.mkdir(exist_ok=True)
 with safe_open(MODEL/'model.safetensors',framework='pt') as f:h=f.get_tensor('model.language_model.embed_tokens.weight')[248045].float().reshape(1,1,1024)
 conv=torch.zeros(1,6144,4);state=torch.zeros(1,16,128,128)
 class Cache:
  def __init__(self,c,s):self.layers=[SimpleNamespace(conv_states=c.clone(),recurrent_states=s.clone())]
  def has_previous_state(self,i):return True
  def update_recurrent_state(self,s,i):self.layers[i].recurrent_states=s
 cache=Cache(conv,state)
 with torch.no_grad():
  ref=l(h,position_embeddings=None,past_key_values=cache);ys=m(h,conv,state)
  diffs={'hidden':float((ys[0]-ref).abs().max()),'conv':float((ys[1]-cache.layers[0].conv_states).abs().max()),'state':float((ys[2]-cache.layers[0].recurrent_states).abs().max())}
  assert max(diffs.values())<1e-5,diffs
  # also nonzero cached states from previous valid step
  cache=Cache(ys[1],ys[2]);ref=l(h,position_embeddings=None,past_key_values=cache);zs=m(h,ys[1],ys[2]);diffs['second_step']=float((zs[0]-ref).abs().max());assert diffs['second_step']<1e-5
 xs=(h,conv,state);names=['hidden','conv','state']
 torch.onnx.export(m,xs,str(p/'layer0.onnx'),input_names=names,output_names=['hidden_out','conv_out','state_out'],opset_version=13,dynamo=False)
 for n,x in zip(names,xs):np.save(p/(n+'.npy'),x.numpy())
 (p/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in names)+'\n')
 np.concatenate([x.numpy().ravel() for x in xs]).tofile(p/'input.f32');np.concatenate([x.numpy().ravel() for x in ys]).tofile(p/'reference.f32')
 (p/'reference-check.json').write_text(json.dumps({'official_transformers':'5.7.0','max_abs':diffs,'scope':'real layer0 and embedding, first/second cached step equivalence'},indent=2));print(diffs,flush=True)

class FullLayer(torch.nn.Module):
 def __init__(self,layer):super().__init__();self.layer=layer
 def forward(self,hidden,past_k,past_v,mask,cos,sin):
  l=self.layer;a=l.self_attn;x=stable_norm(hidden,l.input_layernorm.weight)
  q,gate=a.q_proj(x).reshape(1,1,8,512).chunk(2,dim=-1)
  q=stable_norm(q,a.q_norm.weight).transpose(1,2);k=stable_norm(a.k_proj(x).reshape(1,1,2,256),a.k_norm.weight).transpose(1,2);v=a.v_proj(x).reshape(1,1,2,256).transpose(1,2)
  def rope(t):
   rot=t[:,:,:,:64];rot2=torch.cat([-rot[:,:,:,32:],rot[:,:,:,:32]],dim=-1)
   return torch.cat([rot*cos+rot2*sin,t[:,:,:,64:]],dim=-1)
  q=rope(q);k=rope(k)
  keys=torch.cat([past_k,k],dim=2).repeat_interleave(4,dim=1);values=torch.cat([past_v,v],dim=2).repeat_interleave(4,dim=1)
  scores=(q@keys.transpose(-1,-2))*(256**-0.5)+mask
  attn=torch.softmax(scores,dim=-1)@values
  out=attn.transpose(1,2).reshape(1,1,2048)*torch.sigmoid(gate.reshape(1,1,2048))
  h=hidden+a.o_proj(out)
  return h+l.mlp(stable_norm(h,l.post_attention_layernorm.weight)),k,v

class Head(torch.nn.Module):
 def __init__(self,weight,norm):
  super().__init__();self.register_buffer('weight',weight);self.register_buffer('norm',norm)
 def forward(self,x):
  x=stable_norm(x,self.norm)
  return torch.nn.functional.linear(x,self.weight)
