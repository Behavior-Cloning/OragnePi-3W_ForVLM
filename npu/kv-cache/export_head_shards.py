"""Shard a freshly exported complete ONNX head, preserving every vocabulary row."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
m=json.loads((a.source/'trained-manifest.json').read_text());count=len(m['layers']);dim=m['dim'];width=m['vocab_size']
original=a.source/f'layer{count}.onnx';backup=a.source/'head-full.onnx'
if backup.exists():raise FileExistsError(backup)
original.rename(backup)
model=onnx.load(backup);weights=[t for t in model.graph.initializer if list(t.dims)==[dim,width]]
assert len(weights)==1,[(t.name,list(t.dims)) for t in model.graph.initializer]
weight=weights[0];matrix=numpy_helper.to_array(weight).copy()
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
hidden=np.fromfile(a.source/'case0-hidden.f32',np.float32).reshape(-1,dim)[[0,8,-1]]
expected=np.fromfile(a.source/'case0-logits.f32',np.float32).reshape(-1,width)[[0,8,-1]]
shards=[];observed=[]
for shard,start in enumerate(range(0,width,49152)):
    end=min(start+49152,width);weight.CopyFrom(numpy_helper.from_array(matrix[:,start:end].copy(),weight.name))
    model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value=end-start
    path=a.source/f'layer{count+shard}.onnx';onnx.checker.check_model(model);onnx.save(model,path)
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    observed.append(np.concatenate([session.run(None,{'hidden':x[None,None]})[0].reshape(1,-1) for x in hidden]))
    if shard:shutil.copyfile(a.source/f'layer{count}-dataset.txt',a.source/f'layer{count+shard}-dataset.txt')
    shards.append({'layer':count+shard,'vocab_start':start,'vocab_count':end-start})
actual=np.concatenate(observed,axis=1);np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-4)
relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert relative<1e-4
m['head_shards']=shards;m['head_sharding_reason']='retain all 151674 rows within the SDK output-width limit'
(a.source/'trained-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(a.source/'head-shard-validation.json').write_text(json.dumps({'status':'PASS','relative_l2':relative,'max_abs_error':float(np.abs(actual-expected).max()),'tested_positions':[0,8,m['tokens']-1],'shards':shards},indent=2)+'\n')
print('SHARDED_FULL_VOCAB_PASS',relative)
