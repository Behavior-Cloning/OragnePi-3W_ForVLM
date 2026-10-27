from pathlib import Path
import torch,numpy as np,onnx,subprocess,json,fcntl,time,hashlib,os
from model import Block
from endpoints import Stem,Projector
P=Path(__file__).resolve().parent
S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vit112-native16-r1');S.mkdir(exist_ok=False)
state={'state':'running','scratch':str(S),'pid':os.getpid(),'cases':[]}
def save(): (P/'build-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
xs=[]
for i in [1,2]:
 a=np.fromfile(P.parent/'fixtures'/f'image{i}-112.f32',np.float32).reshape(112,112,3).transpose(2,0,1)[None].copy();xs.append(torch.from_numpy(a));a.tofile(S/f'input{i}.f32')
try:
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for name,model in [('stem',Stem())]+[(f'layer{i:02d}',Block(i)) for i in range(24)]+[('projector',Projector())]:
   state['current']=name;save();src=S/name;src.mkdir();model.eval()
   for i,x in enumerate(xs,1):np.save(src/f'input{i}.npy',x.numpy())
   torch.onnx.export(model,xs[0],str(src/'layer0.onnx'),input_names=['hidden'] if name.startswith('layer') else ['input'],output_names=['hidden_out'] if name.startswith('layer') else ['output'],opset_version=13,dynamo=False)
   onnx.checker.check_model(str(src/'layer0.onnx'));(src/'layer0-dataset.txt').write_text('input1.npy\ninput2.npy\n')
   with torch.no_grad(): xs=[model(x) for x in xs]
   for i,x in enumerate(xs,1):x.numpy().tofile(src/f'reference{i}.f32')
   dest=S/(name+'-compiled')
   cmd=['/opt/opi-lab/export-venv/bin/python',str(P/'compile_selective.py'),'--source',str(src),'--results-root',str(S),'--output',str(dest),'--dtype','int16','--timeout','240']
   with (P/(name+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1050)
   assert z.returncode==0,(name,z.returncode)
   nb=dest/'export_nbg_unify/network_binary.nb';assert nb.is_file()
   state['cases'].append({'name':name,'path':str(nb),'sha256':hashlib.sha256(nb.read_bytes()).hexdigest(),'bytes':nb.stat().st_size});save();print('compiled',name,flush=True)
  state['state']='compiled-not-board-validated';save()
except BaseException as e:
 state['state']='failed';state['error']=str(e);save();raise
