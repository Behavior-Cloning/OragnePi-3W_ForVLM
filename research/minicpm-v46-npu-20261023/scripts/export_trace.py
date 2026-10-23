from model_layers import *
import subprocess
class Trace(torch.nn.Module):
 def __init__(self,l):super().__init__();self.l=l
 def forward(self,h,conv,state):
  l=self.l;a=l.linear_attn;x=stable_norm(h,l.input_layernorm.weight);qkv=a.in_proj_qkv(x).transpose(1,2);nc=torch.cat([conv[:,:,1:],qkv],-1)
  mixed=torch.nn.functional.silu(sum(nc[:,:,j]*a.conv1d.weight[:,0,j] for j in range(4)))
  q,k,v=[t.reshape(1,16,1,128) for t in mixed.split(2048,dim=-1)];q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*(128**-.5);k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
  g=-a.A_log.exp().reshape(1,16,1,1)*torch.nn.functional.softplus(a.in_proj_a(x)+a.dt_bias).reshape(1,16,1,1);beta=a.in_proj_b(x).sigmoid().reshape(1,16,1,1);ns=state*g.exp();ns=ns+k.transpose(-1,-2)@((v-k@ns)*beta)
  core=(q@ns).reshape(16,128);z=a.in_proj_z(x).reshape(16,128);gn=stable_norm(core,a.norm.weight,False)*torch.nn.functional.silu(z);res=h+a.out_proj(gn.reshape(1,1,2048));pn=stable_norm(res,l.post_attention_layernorm.weight);mo=l.mlp(pn)
  return x,core,gn.reshape(1,16,128),res,pn,mo,res+mo
l=load_layer(0);m=Trace(l).eval();torch.set_grad_enabled(False);torch.set_num_threads(4);p=ROOT/'trace-linear0';p.mkdir(exist_ok=True);names=['hidden','conv','state'];xs=tuple(torch.from_numpy(np.load(ROOT/'linear0-stable'/(n+'.npy'))) for n in names);ys=m(*xs);outs=['norm_input','core','gated_norm','residual','post_norm','mlp','hidden']
torch.onnx.export(m,xs,str(p/'layer0.onnx'),input_names=names,output_names=[n+'_out' for n in outs],opset_version=13,dynamo=False)
for n,x in zip(names,xs):np.save(p/(n+'.npy'),x.numpy())
(p/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in names)+'\n');np.concatenate([x.numpy().ravel() for x in xs]).tofile(p/'input.f32');np.concatenate([x.numpy().ravel() for x in ys]).tofile(p/'reference.f32');(p/'outputs.json').write_text(json.dumps([{'name':n,'count':y.numel(),'rms':float(y.square().mean().sqrt()),'min':float(y.min()),'max':float(y.max())} for n,y in zip(outs,ys)],indent=2))
r=Path(__file__).resolve().parents[1]
cmd=['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(p),'--results-root',str(ROOT),'--output',str(ROOT/'trace-linear0-fp16'),'--dtype','fp16','--timeout','240']
with (r/'results/trace-compile.log').open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
print('COMPILE',z.returncode,flush=True)
