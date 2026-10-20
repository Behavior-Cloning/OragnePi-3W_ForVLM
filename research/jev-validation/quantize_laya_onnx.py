from pathlib import Path
import onnx,json,hashlib,os
from onnxruntime.quantization import quantize_dynamic,QuantType
r=Path(__file__).resolve().parent;m=onnx.load(r/'models/laya/laya-fp32.onnx');init={x.name for x in m.graph.initializer}
names=[x.name for x in m.graph.node if x.op_type=='MatMul' and x.name.startswith('/encoder/') and len(x.input)>1 and x.input[1] in init]
assert len(names)>0
del m.graph.value_info[:]
out=r/'models/laya/laya-int8.onnx'
quantize_dynamic(m,str(out),nodes_to_quantize=names,op_types_to_quantize=['MatMul'],weight_type=QuantType.QInt8,per_channel=False)
del m
model=onnx.load(out);q=[n.name for n in model.graph.node if n.op_type=='MatMulInteger'];assert len(q)==len(names);assert all(n.startswith('/encoder/') for n in q)
(r/'results/laya-onnx-int8-export.json').write_text(json.dumps(dict(status='exported_not_quality_verified',scope='encoder constant-weight MatMul only; head FP32',per_channel=False,selected_nodes=names,actual_int8_nodes=q,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2));print('exported',len(q),out.stat().st_size,flush=True)
