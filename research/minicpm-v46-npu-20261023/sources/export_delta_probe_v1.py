import torch,numpy as np,json,hashlib
from pathlib import Path
r=Path('/opt/opi-lab/minicpm-v46-npu-20261023');torch.set_num_threads(2);torch.manual_seed(46)
class Delta(torch.nn.Module):
 def __init__(self,kind):super().__init__();self.kind=kind
 def forward(self,q,k,v,g,beta,state):
  q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*(128**-0.5)
  k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
  decayed=state*torch.exp(g)
  if self.kind=='reduce':
   mem=(decayed*k.transpose(-1,-2)).sum(-2,keepdim=True)
   updated=decayed+k.transpose(-1,-2)*((v-mem)*beta)
   out=(updated*q.transpose(-1,-2)).sum(-2,keepdim=True)
  else:
   mem=k@decayed
   updated=decayed+k.transpose(-1,-2)@((v-mem)*beta)
   out=q@updated
  return out,updated
for kind in ['reduce','matmul']:
 p=r/kind;p.mkdir(exist_ok=True)
 xs=(torch.randn(16,1,128),torch.randn(16,1,128),torch.randn(16,1,128),-torch.rand(16,1,1),torch.rand(16,1,1),torch.randn(16,128,128)*.1)
 names=['q','k','v','g','beta','state'];m=Delta(kind).eval()
 torch.onnx.export(m,xs,str(p/'layer0.onnx'),input_names=names,output_names=['out','state_out'],opset_version=13,dynamo=False)
 with torch.no_grad():ys=m(*xs)
 np.concatenate([x.numpy().ravel() for x in xs]).tofile(p/'input.f32');np.concatenate([y.numpy().ravel() for y in ys]).tofile(p/'reference.f32')
 lines=[]
 for n,x in zip(names,xs):np.save(p/(n+'.npy'),x.numpy());lines.append(n+'.npy')
 (p/'layer0-dataset.txt').write_text(' '.join(lines)+'\n')
 (p/'fixture.json').write_text(json.dumps({'seed':46,'kind':kind,'model_dimensions':'16 heads x 128 key x 128 value','inputs':{n:list(x.shape) for n,x in zip(names,xs)},'outputs':[list(y.shape) for y in ys],'scope':'synthetic one-step core, not full model nor language quality'},indent=2))
 print(kind,flush=True)
