from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,torch.nn.functional as F,onnx
sys.path.insert(0,'/opt/opi-lab/builds/intern-frontmerge-20261027-r1/source/gguf-py')
from gguf import GGUFReader
R=Path(__file__).resolve().parent;D=R.parent;torch.set_num_threads(4)
source=Path('/mnt/f/Files/OPi-Lab/models/internvl3-1b-vision/mmproj-InternVL3-1B-Instruct-f16.gguf')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='89e15ad85554c6a29910187c4f9689d992e8a7b586941cef68651bf91b6bc4b8'
g=GGUFReader(str(source));weights={t.name:torch.from_numpy(t.data.copy().astype(np.float32)) for t in g.tensors if True}
class Block(torch.nn.Module):
 def __init__(self,layer):
  super().__init__()
  for k,v in weights.items():
   if k.startswith('v.blk.%d.'%layer):self.register_buffer(('v.blk.0.'+k.split('.',3)[3]).replace('.','_'),v)
 def w(self,k):return getattr(self,'v_blk_0_'+k.replace('.','_'))
 def norm(self,x,k):
  mean=x.mean(-1,keepdim=True);center=x-mean
  return center*torch.rsqrt((center*center).mean(-1,keepdim=True)+1e-6)*self.w(k+'.weight')+self.w(k+'.bias')
 def linear(self,x,k):return F.linear(x,self.w(k+'.weight'),self.w(k+'.bias'))
 def forward(self,x):
  z=self.norm(x,'ln1');q,k,v=[self.linear(z,'attn_'+name).reshape(1,-1,16,64).transpose(1,2) for name in ['q','k','v']]
  a=torch.softmax((q@k.transpose(-1,-2))*0.125,dim=-1)@v
  a=a.transpose(1,2).reshape(1,-1,1024)
  x=x+self.linear(a,'attn_out')*self.w('ls1.weight')
  z=self.linear(self.norm(x,'ln2'),'ffn_up')
  # GGML GELU uses tanh approximation; this probe matches that convention.
  z=F.gelu(z,approximate='tanh')
  return x+self.linear(z,'ffn_down')*self.w('ls2.weight')
