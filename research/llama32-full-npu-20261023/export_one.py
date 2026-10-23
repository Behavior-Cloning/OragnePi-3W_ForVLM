"""Export exactly one real decoder layer or full-vocabulary shard."""
from model import *
import argparse,onnx,onnxruntime as ort,shutil
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--stage',type=Path,required=True);a=p.parse_args();assert 0<=a.index<20
assert shutil.disk_usage('/mnt/f').free>8*1024**3
torch.set_num_threads(2);reader,tensors=load();a.stage.mkdir(parents=True,exist_ok=True);head=a.index>=16
model=HeadShard(tensors,(a.index-16)*32064,(a.index-15)*32064).eval() if head else decoder(reader,a.index)
cal=BUILD/'calibration'/('head' if head else f'layer{a.index}');rows=(cal/'dataset.txt').read_text().splitlines();names=['hidden'] if head else INPUTS;outs=['logits'] if head else OUTPUTS
feed={n:np.load(cal/f) for n,f in zip(names,rows[0].split())};path=a.stage/f'layer{a.index}.onnx';torch.onnx.export(model,tuple(torch.from_numpy(v) for v in feed.values()),str(path),input_names=names,output_names=outs,opset_version=13,dynamo=False);g=onnx.load(path);g.ir_version=8;onnx.checker.check_model(g,full_check=True);onnx.save(g,path)
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;sess=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']);checks=[]
for row in rows:
 feed={n:np.load(cal/f) for n,f in zip(names,row.split())}
 with torch.inference_mode():ref=model(*[torch.from_numpy(v) for v in feed.values()]);ref=[ref] if head else list(ref)
 ys=sess.run(outs,feed)
 for name,y,z in zip(outs,ys,ref):
  z=z.numpy();rel=float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-12));assert rel<3e-5,(a.index,name,rel);checks.append(dict(output=name,relative_l2=rel,max_abs=float(np.max(np.abs(y-z)))))
# Absolute host paths become /work paths by the importer; use relative traversal here.
import os
(a.stage/f'layer{a.index}-dataset.txt').write_text('\n'.join(' '.join(os.path.relpath(cal/f,a.stage) for f in row.split()) for row in rows)+'\n')
manifest=dict(index=a.index,kind='head' if head else 'decoder',checks=checks,onnx_sha256=sha(path),onnx_bytes=path.stat().st_size,actual_layer_calibration=str(cal),full_vocabulary_range=[(a.index-16)*32064,(a.index-15)*32064] if head else None)
(a.stage/'export-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
