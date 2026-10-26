from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,torch.nn.functional as F,onnx
sys.path.insert(0,'/opt/opi-lab/builds/intern-frontmerge-20261027-r1/source/gguf-py')
from gguf import GGUFReader
R=Path(__file__).resolve().parent;D=R.parent;torch.set_num_threads(4)
source=Path('/mnt/f/Files/OPi-Lab/models/internvl3-1b-vision/mmproj-InternVL3-1B-Instruct-f16.gguf')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='89e15ad85554c6a29910187c4f9689d992e8a7b586941cef68651bf91b6bc4b8'
g=GGUFReader(str(source));weights={t.name:torch.from_numpy(t.data.copy().astype(np.float32)) for t in g.tensors if t.name.startswith('v.blk.0.') or t.name in ['v.patch_embd.weight','v.patch_embd.bias','v.class_embd','v.position_embd.weight']}
class Block(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for k,v in weights.items():
   if k.startswith('v.blk.0.'):self.register_buffer(k.replace('.','_'),v)
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
model=Block().eval();records=[]
for side in [112,224]:
 out=R/('s'+str(side));out.mkdir(exist_ok=False);inputs=[]
 grid=side//14
 pos=weights['v.position_embd.weight'];patchpos=pos[:,:1024].reshape(1,32,32,1024).permute(0,3,1,2)
 patchpos=F.interpolate(patchpos,size=(grid,grid),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,grid*grid,1024)
 for i in [1,2]:
  pix=torch.from_numpy(np.fromfile(D/'fixtures'/('image%d-%d.f32'%(i,side)),np.float32).reshape(side,side,3).copy()).permute(2,0,1)[None]
  stem=F.conv2d(pix,weights['v.patch_embd.weight'],weights['v.patch_embd.bias'],stride=14).flatten(2).transpose(1,2)
  x=torch.cat([stem,weights['v.class_embd']],dim=1)+torch.cat([patchpos,pos[:,1024:]],dim=1)
  with torch.no_grad():y=model(x)
  np.save(out/('input%d.npy'%i),x.numpy());y.numpy().tofile(out/('output%d.f32'%i));inputs.append(x)
 torch.onnx.export(model,inputs[0],str(out/'layer0.onnx'),input_names=['hidden'],output_names=['hidden_out'],opset_version=13,dynamo=False)
 onnx.checker.check_model(str(out/'layer0.onnx'))
 (out/'layer0-dataset.txt').write_text('input1.npy\ninput2.npy\n')
 records.append({'side':side,'patch_tokens':grid*grid,'including_cls':grid*grid+1,'onnx_sha256':hashlib.sha256((out/'layer0.onnx').read_bytes()).hexdigest(),'input_shape':list(inputs[0].shape)})
(R/'export.json').write_text(json.dumps({'status':'exported_not_board_validated','scope':'first real ViT block; 2 development-image calibration inputs; not whole tower or sufficient final calibration','GELU':'tanh approximation matching GGML path, not proof of HF exact-GELU parity','cases':records},indent=2)+'\n');print(records)
