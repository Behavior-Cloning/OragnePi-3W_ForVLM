from model_layers import *
import sys,subprocess
class Pre(torch.nn.Module):
 def __init__(self,l):super().__init__();self.l=l
 def forward(self,hidden,conv):
  a=self.l.linear_attn;x=self.l.input_layernorm(hidden);qkv=a.in_proj_qkv(x).transpose(1,2);new_conv=torch.cat([conv[:,:,1:],qkv],dim=-1)
  mixed=torch.nn.functional.silu((new_conv*a.conv1d.weight[:,0,:]).sum(-1));q,k,v=[t.reshape(1,16,1,128) for t in mixed.split(2048,dim=-1)]
  z=a.in_proj_z(x).reshape(16,128);g=-a.A_log.exp().reshape(1,16,1,1)*torch.nn.functional.softplus(a.in_proj_a(x)+a.dt_bias).reshape(1,16,1,1);beta=a.in_proj_b(x).sigmoid().reshape(1,16,1,1)
  return q,k,v,g,beta,z,new_conv
class Post(torch.nn.Module):
 def __init__(self,l):super().__init__();self.l=l
 def forward(self,hidden,out,z):
  l=self.l;a=l.linear_attn;out=a.norm(out.reshape(16,128),z).reshape(1,1,2048);h=hidden+a.out_proj(out);return h+l.mlp(l.post_attention_layernorm(h))
class Conv(torch.nn.Module):
 def __init__(self,l):super().__init__();self.register_buffer('w',l.linear_attn.conv1d.weight[:,0,:])
 def forward(self,x):return torch.nn.functional.silu((x*self.w).sum(-1))
torch.set_num_threads(4);torch.set_grad_enabled(False);l=load_layer(0)
h=torch.from_numpy(np.load(ROOT/'linear0/hidden.npy'));conv=torch.from_numpy(np.load(ROOT/'linear0/conv.npy'));state=torch.from_numpy(np.load(ROOT/'linear0/state.npy'))
pre=Pre(l).eval();xs=pre(h,conv);q,k,v,g,beta,z,newconv=xs
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
out,_=torch_recurrent_gated_delta_rule(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),g.reshape(1,1,16),beta.reshape(1,1,16),state,True,True)
for name,m,args,names in [('pre',pre,(h,conv),['hidden','conv']),('post',Post(l).eval(),(h,out,z),['hidden','out','z']),('conv',Conv(l).eval(),(newconv,),['x'])]:
 p=ROOT/('split-'+name);p.mkdir(exist_ok=True);torch.onnx.export(m,args,str(p/'layer0.onnx'),input_names=names,opset_version=13,dynamo=False)
 for n,x in zip(names,args):np.save(p/(n+'.npy'),x.numpy())
 (p/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in names)+'\n');print('export',name,flush=True)
r=Path(__file__).resolve().parents[1]
for kind in ['split-pre','split-post','split-conv']:
 cmd=['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(ROOT/kind),'--results-root',str(ROOT),'--output',str(ROOT/(kind+'-fp16')),'--dtype','fp16','--timeout','180']
 with (r/'results'/(kind+'-compile.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('compile',kind,z.returncode,flush=True)
