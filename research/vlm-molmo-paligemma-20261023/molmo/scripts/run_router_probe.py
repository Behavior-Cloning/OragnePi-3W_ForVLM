import torch,numpy as np,json,hashlib,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.manual_seed(20261023)
class Router(torch.nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.gate=torch.nn.Linear(2048,64,bias=False)
 def forward(self,x):
  probs=torch.softmax(self.gate(x).float(),dim=-1)
  if self.kind=='softmax-control':return probs
  return torch.topk(probs,8,dim=-1)
for kind in ['topk-router','softmax-control']:
 p=r/'probe'/kind;p.mkdir(parents=True,exist_ok=True);m=Router(kind).eval();x=torch.randn(1,2048)
 outputs=['values','indices'] if kind=='topk-router' else ['probabilities']
 torch.onnx.export(m,(x,),str(p/'layer0.onnx'),input_names=['hidden'],output_names=outputs,opset_version=13,dynamo=False)
 np.save(p/'hidden.npy',x.numpy());(p/'layer0-dataset.txt').write_text('hidden.npy\n')
 with torch.no_grad():y=m(x)
 if isinstance(y,torch.Tensor):y=(y,)
 for name,val in zip(outputs,y):np.save(p/(name+'-reference.npy'),val.numpy())
 (p/'fixture.json').write_text(json.dumps({'scope':'Synthetic random gate, exact MolmoE D2048/64 experts/top8 decode router shapes. Operator compiler test only; no pretrained weights, no expert dispatch, no full model, no board latency or quality.','seed':20261023,'config_sha256':hashlib.sha256((r/'source/config.json').read_bytes()).hexdigest(),'torch':torch.__version__,'shape':[1,2048],'outputs':{n:list(v.shape) for n,v in zip(outputs,y)},'onnx_sha256':hashlib.sha256((p/'layer0.onnx').read_bytes()).hexdigest()},indent=2))
 command=[sys.executable,str(r/'scripts/compile_probe.py'),'--source',str(p),'--results-root',str(r),'--output',str(r/'compiled'/kind),'--dtype','fp16','--timeout','180']
 start=time.monotonic();rc=subprocess.run(command).returncode;print('PROBE',kind,rc,time.monotonic()-start,flush=True)
