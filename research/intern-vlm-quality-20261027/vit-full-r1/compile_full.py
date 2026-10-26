from pathlib import Path
import os,sys,subprocess,json,time,shutil,hashlib,fcntl
import torch,numpy as np,onnx
from model import Block
P=Path(__file__).resolve().parent;R=P.parents[2];scratch=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vitfull-20261027-r1');scratch.mkdir(exist_ok=False)
state=P/'compile-full-status.json';assert not state.exists();out=P/'int16-nbg';out.mkdir(exist_ok=False)
s={'status':'starting','pid':os.getpid(),'scratch':str(scratch),'side':224,'layers_total':24,'cases':[],'started':time.time()}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for layer in range(24):
  s.update(status='exporting',layer=layer);save();source=scratch/('l%02d'%layer);source.mkdir()
  for i in [1,2]:shutil.copy2(P/('s224/l%02d-input%d.npy'%(layer,i)),source/('input%d.npy'%i))
  model=Block(layer).eval();x=torch.from_numpy(np.load(source/'input1.npy'))
  torch.onnx.export(model,x,str(source/('layer%d.onnx'%layer)),input_names=['hidden'],output_names=['hidden_out'],opset_version=13,dynamo=False);onnx.checker.check_model(str(source/('layer%d.onnx'%layer)))
  (source/('layer%d-dataset.txt'%layer)).write_text('input1.npy\ninput2.npy\n')
  dest=scratch/('compiled%02d'%layer);s.update(status='compiling');save()
  cmd=['/opt/opi-lab/export-venv/bin/python',str(R/'npu/kv-cache/compile_probe.py'),'--source',str(source),'--results-root',str(scratch),'--output',str(dest),'--layer',str(layer),'--dtype','int16','--timeout','240']
  with (P/('compile-l%02d.log'%layer)).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1050)
  if z.returncode:
   s.update(status='failed',returncode=z.returncode);save();raise RuntimeError('compile layer '+str(layer))
  nb=dest/'export_nbg_unify/network_binary.nb';shutil.copy2(nb,out/('layer%02d.nb'%layer))
  meta=out/('layer%02d-conversion.json'%layer);shutil.copy2(dest/'conversion.json',meta)
  s['cases'].append({'layer':layer,'onnx_sha256':hashlib.sha256((source/('layer%d.onnx'%layer)).read_bytes()).hexdigest(),'nb_sha256':hashlib.sha256(nb.read_bytes()).hexdigest(),'nb_bytes':nb.stat().st_size,'scratch':str(dest),'source':str(source)});save();print('compiled',layer,nb.stat().st_size,flush=True)
 s.update(status='compiled_not_board_validated',ended=time.time());save()
