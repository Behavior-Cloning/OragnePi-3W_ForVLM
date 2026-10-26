from pathlib import Path
import torch,torch.nn.functional as F,numpy as np,json,subprocess,fcntl,shutil,hashlib
from model import weights
P=Path(__file__).resolve().parent;R=P.parents[2];scratch=Path(json.loads((P/'compile-full-status.json').read_text())['scratch']);out=P/'endpoints-nbg';out.mkdir(exist_ok=False)
class Stem(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for n,k in [('w','v.patch_embd.weight'),('b','v.patch_embd.bias'),('cls','v.class_embd')]:self.register_buffer(n,weights[k])
  pos=weights['v.position_embd.weight'];patch=pos[:,:1024].reshape(1,32,32,1024).permute(0,3,1,2)
  patch=F.interpolate(patch,size=(16,16),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,256,1024)
  self.register_buffer('pos',torch.cat([patch,pos[:,1024:]],dim=1))
 def forward(self,x):
  x=F.conv2d(x,self.w,self.b,stride=14).flatten(2).transpose(1,2)
  return torch.cat([x,self.cls],dim=1)+self.pos
class Projector(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for n,k in [('nw','0.weight'),('nb','0.bias'),('w1','1.weight'),('b1','1.bias'),('w3','3.weight'),('b3','3.bias')]:self.register_buffer(n,weights['mm.model.mlp.'+k])
 def forward(self,x):
  x=x[:,:256,:].reshape(1,16,8,2048).permute(0,2,1,3).contiguous().reshape(1,8,8,4096).permute(0,2,1,3).contiguous().reshape(1,64,4096)
  mean=x.mean(-1,keepdim=True);c=x-mean;x=c*torch.rsqrt((c*c).mean(-1,keepdim=True)+1e-5)*self.nw+self.nb
  return F.linear(F.gelu(F.linear(x,self.w1,self.b1),approximate='tanh'),self.w3,self.b3)
records=[]
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name,model in [('stem',Stem().eval()),('projector',Projector().eval())]:
  source=scratch/name;source.mkdir();examples=[]
  for i in [1,2]:
   if name=='stem':a=np.fromfile(P.parent/'fixtures'/('image%d-224.f32'%i),np.float32).reshape(224,224,3).transpose(2,0,1)[None].copy()
   else:a=np.fromfile(P/('s224/l23-output%d.f32'%i),np.float32).reshape(1,257,1024)
   x=torch.from_numpy(a);np.save(source/('input%d.npy'%i),a);examples.append(x)
   with torch.no_grad():y=model(x).numpy()
   y.tofile(out/(name+'-reference%d.f32'%i))
  torch.onnx.export(model,examples[0],str(source/'layer0.onnx'),input_names=['input'],output_names=['output'],opset_version=13,dynamo=False)
  (source/'layer0-dataset.txt').write_text('input1.npy\ninput2.npy\n');dest=scratch/(name+'-compiled')
  cmd=['/opt/opi-lab/export-venv/bin/python',str(R/'npu/kv-cache/compile_probe.py'),'--source',str(source),'--results-root',str(scratch),'--output',str(dest),'--dtype','int16','--timeout','240']
  with (P/(name+'-compile.log')).open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1050)
  assert z.returncode==0,name
  nb=dest/'export_nbg_unify/network_binary.nb';shutil.copy2(nb,out/(name+'.nb'));records.append({'name':name,'nb_sha256':hashlib.sha256(nb.read_bytes()).hexdigest(),'bytes':nb.stat().st_size,'source':str(source),'compiled':str(dest)});print(name,records[-1],flush=True)
  (P/'endpoints-status.json').write_text(json.dumps({'status':'compiled_not_board_validated' if len(records)==2 else 'running','cases':records},indent=2)+'\n')
