import torch,json,hashlib,time
from pathlib import Path
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from onnxruntime.quantization import quantize_dynamic,QuantType
import onnxruntime as ort
r=Path(__file__).resolve().parent;d=r/'models/nano33';torch.set_num_threads(2)
m=AutoModelForSequenceClassification.from_pretrained(d,local_files_only=True,attn_implementation='eager').eval();tok=AutoTokenizer.from_pretrained(d,local_files_only=True)
class Wrapper(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,input_ids,attention_mask,token_type_ids):return self.m(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).logits
x=tok(['question: Is the claim supported? option: yes','question: Is the claim supported? option: no'],['The sky is blue.']*2,padding=True,return_tensors='pt')
args=tuple(x[k] for k in ['input_ids','attention_mask','token_type_ids'])
torch.onnx.export(Wrapper(m).eval(),args,str(d/'nano33-fp32.onnx'),input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],dynamic_axes={k:{0:'batch',1:'sequence'} for k in x}|{'logits':{0:'batch'}},opset_version=17,dynamo=False)
quantize_dynamic(str(d/'nano33-fp32.onnx'),str(d/'nano33-int8.onnx'),per_channel=True,weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])
m.eval()
with torch.inference_mode():ref=m(**x).logits.numpy()
results={}
for name in ['nano33-fp32.onnx','nano33-int8.onnx']:
 so=ort.SessionOptions();so.intra_op_num_threads=2;s=ort.InferenceSession(str(d/name),so,providers=['CPUExecutionProvider']);z=s.run(None,{k:v.numpy() for k,v in x.items()})[0];results[name]={'bytes':(d/name).stat().st_size,'sha256':hashlib.sha256((d/name).read_bytes()).hexdigest(),'smoke_max_abs_logit_error':float(abs(z-ref).max()),'torch_logits':ref.tolist(),'onnx_logits':z.tolist()}
(r/'results/nano-onnx-export.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
