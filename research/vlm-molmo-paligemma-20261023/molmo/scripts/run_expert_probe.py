from pathlib import Path
import torch,numpy as np,json,hashlib,subprocess,sys
r=Path(__file__).resolve().parents[1];torch.manual_seed(20261023);torch.set_num_threads(2)
class Expert(torch.nn.Module):
 def __init__(self,dynamic):
  super().__init__();self.dynamic=dynamic;self.gate=torch.nn.Linear(2048,1024,bias=False);self.up=torch.nn.Linear(2048,1024,bias=False);self.down=torch.nn.Linear(1024,2048,bias=False)
 def forward(self,x,mask,weights):
  if self.dynamic:
   idx,top_x=torch.where(mask);s=x[None,top_x].reshape(-1,2048);w=weights[top_x,idx,None]
  else:s=x;w=weights[:,:1]
  return self.down(torch.nn.functional.silu(self.gate(s))*self.up(s))*w
for dynamic in [True,False]:
 name='dynamic-one-expert' if dynamic else 'static-one-expert';p=r/'probe'/name;p.mkdir(parents=True,exist_ok=True);m=Expert(dynamic).eval()
 x=torch.randn(1,2048);mask=torch.zeros(8,1,dtype=torch.int64);mask[0,0]=1;weights=torch.softmax(torch.randn(1,8),dim=-1)
 xs=(x,mask,weights);names=['hidden','expert_mask','routing_weights']
 torch.onnx.export(m,xs,str(p/'layer0.onnx'),input_names=names,output_names=['expert_output'],opset_version=13,dynamo=False)
 import onnx
 graph=onnx.load(str(p/'layer0.onnx'));actual=[i.name for i in graph.graph.input]
 for n,v in zip(names,xs):np.save(p/(n+'.npy'),v.numpy())
 (p/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in actual)+'\n')
 with torch.no_grad():np.save(p/'output-reference.npy',m(*xs).numpy())
 (p/'fixture.json').write_text(json.dumps({'scope':'Random weights, exact one-expert D2048/FF1024 shape from MolmoeMLP. Dynamic branch mirrors torch.where and gather dispatch for one of 64 experts. No full model, pretrained weights or actual quality claim.','dynamic':dynamic,'seed':20261023,'ops':sorted(set(n.op_type for n in graph.graph.node)),'source_sha256':hashlib.sha256((r/'source/modeling_molmoe.py').read_bytes()).hexdigest()},indent=2))
 command=[sys.executable,str(r/'scripts/compile_probe.py'),'--source',str(p),'--results-root',str(r),'--output',str(r/'compiled'/name),'--dtype','fp16','--timeout','120']
 result=subprocess.run(command);print('PROBE',name,result.returncode,flush=True)
