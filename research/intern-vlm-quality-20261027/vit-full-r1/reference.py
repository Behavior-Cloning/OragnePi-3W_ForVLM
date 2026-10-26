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

records=[]
for side in [112,224]:
 out=R/('s'+str(side));out.mkdir(exist_ok=False);grid=side//14
 pos=weights['v.position_embd.weight'];patchpos=pos[:,:1024].reshape(1,32,32,1024).permute(0,3,1,2)
 patchpos=F.interpolate(patchpos,size=(grid,grid),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,grid*grid,1024)
 images=[]
 for i in [1,2]:
  pix=torch.from_numpy(np.fromfile(D/'fixtures'/('image%d-%d.f32'%(i,side)),np.float32).reshape(side,side,3).copy()).permute(2,0,1)[None]
  stem=F.conv2d(pix,weights['v.patch_embd.weight'],weights['v.patch_embd.bias'],stride=14).flatten(2).transpose(1,2)
  x=torch.cat([stem,weights['v.class_embd']],dim=1)+torch.cat([patchpos,pos[:,1024:]],dim=1)
  images.append(x)
 with torch.no_grad():
  for layer in range(24):
   block=Block(layer).eval()
   for i in range(2):
    np.save(out/('l%02d-input%d.npy'%(layer,i+1)),images[i].numpy())
    images[i]=block(images[i]);images[i].numpy().tofile(out/('l%02d-output%d.f32'%(layer,i+1)))
   print('reference',side,layer,flush=True)
  for i,x in enumerate(images,1):
   x=x[:,:-1,:].contiguous()
   if side==224:
    x=x.reshape(1,grid,grid//2,2048).permute(0,2,1,3).contiguous().reshape(1,grid//2,grid//2,4096).permute(0,2,1,3).contiguous().reshape(1,64,4096)
   else:
    # Retains all64 tokens; reproduces existing untrained repeat adapter only.
    x=x.repeat(1,1,4)
   x=F.layer_norm(x,(4096,),weights['mm.model.mlp.0.weight'],weights['mm.model.mlp.0.bias'],1e-5)
   x=F.linear(x,weights['mm.model.mlp.1.weight'],weights['mm.model.mlp.1.bias']);x=F.gelu(x,approximate='tanh');x=F.linear(x,weights['mm.model.mlp.3.weight'],weights['mm.model.mlp.3.bias'])
   x.numpy().tofile(out/('embedding%d.f32'%i))
   ref=D/'vision-frontmerge/board-r4-results/results'/('%s-%d.f32'%('native224' if side==224 else 'keep112',i))
   if not ref.exists():
    refs=list((D/'vision-frontmerge/board-r4-results/results').glob('*112*'));raise RuntimeError(str(ref)+' '+str(refs))
   a=np.fromfile(ref,np.float32).astype('float64');b=x.numpy().reshape(-1).astype('float64');assert len(a)==len(b)==64*896 and np.isfinite(b).all()
   records.append({'side':side,'image':i,'relative_l2_vs_ggml_fp16':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'max_abs':float(abs(a-b).max())})
(R/'reference-audit.json').write_text(json.dumps({'scope':'24-layer real-weight FP32/tanh reference vs frozen GGML FP16 vision;112 repeat adapter still untrained','records':records},indent=2)+'\n');print(json.dumps(records,indent=2))
