import torch
import torch.nn.functional as F
from model import weights
class Stem(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for n,k in [('w','v.patch_embd.weight'),('b','v.patch_embd.bias'),('cls','v.class_embd')]:self.register_buffer(n,weights[k])
  pos=weights['v.position_embd.weight'];patch=pos[:,:1024].reshape(1,32,32,1024).permute(0,3,1,2)
  patch=F.interpolate(patch,size=(8,8),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,64,1024)
  self.register_buffer('pos',torch.cat([patch,pos[:,1024:]],dim=1))
 def forward(self,x):
  x=F.conv2d(x,self.w,self.b,stride=14).flatten(2).transpose(1,2)
  return torch.cat([x,self.cls],dim=1)+self.pos
class Projector(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for n,k in [('nw','0.weight'),('nb','0.bias'),('w1','1.weight'),('b1','1.bias'),('w3','3.weight'),('b3','3.bias')]:self.register_buffer(n,weights['mm.model.mlp.'+k])
 def forward(self,x):
  x=x[:,:64,:].reshape(1,8,4,2048).permute(0,2,1,3).contiguous().reshape(1,4,4,4096).permute(0,2,1,3).contiguous().reshape(1,16,4096)
  mean=x.mean(-1,keepdim=True);c=x-mean;x=c*torch.rsqrt((c*c).mean(-1,keepdim=True)+1e-5)*self.nw+self.nb
  return F.linear(F.gelu(F.linear(x,self.w1,self.b1),approximate='tanh'),self.w3,self.b3)
