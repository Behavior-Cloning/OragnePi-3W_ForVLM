from pathlib import Path
import os,sys,json,gc,time,hashlib
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import torch
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/"sources"));import laya_export_onnx_v0412 as ex
from board_adapters import Adapter
import onnxruntime as ort
factory=ort.InferenceSession
def session(path,*args,**kw):
 so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
 return factory(path,sess_options=so,providers=["CPUExecutionProvider"])
ort.InferenceSession=session
torch.set_num_threads(2)
t=time.time();out=r/"models/laya/laya-fp32.onnx"
m=ex.export_to_onnx(str(r/"models/laya"),str(out),verify=True);del m;gc.collect()
cases=json.loads((r/"cases.json").read_text());actions=json.loads((r/"actions.json").read_text())
inputs=[]
a=Adapter("laya",r,2)
for count in [10,100]:
 for budget in [64,256]:
  state,n=a.state(cases[4],budget);opts={v["id"]:v["description"] for v in actions[:count]};z=a.decide(state,opts)
  inputs.append(dict(count=count,budget=budget,state_tokens=n,state=state,options=opts,eager=z))
del a;gc.collect();os.environ["LAYA_ORT"]="fp32";a=Adapter("laya",r,2)
for x in inputs:
 x["onnx"]=a.decide(x["state"],x["options"]);x["answer_match"]=x["eager"]["answer"]==x["onnx"]["answer"];print("actual-shape",x["count"],x["budget"],x["answer_match"],flush=True)
h=hashlib.sha256()
with out.open("rb") as f:
 for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
(r/"results/laya-onnx-export.json").write_text(json.dumps(dict(status="verified",exporter_blob="d809735d974806a4185e155df6edc83924a59221",seconds=time.time()-t,bytes=out.stat().st_size,sha256=h.hexdigest(),tests=inputs),ensure_ascii=False,indent=2))
