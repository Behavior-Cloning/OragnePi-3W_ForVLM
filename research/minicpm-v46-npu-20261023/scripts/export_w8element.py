from pathlib import Path
import torch,numpy as np,sys,json
from safetensors import safe_open
from model_layers_elementwise import *
torch.set_num_threads(4);torch.set_grad_enabled(False)
for name in sys.argv[1:]:
 src=ROOT/'full-epssafe'/name;o=ROOT/'w8element-source'/name;o.mkdir(parents=True,exist_ok=True)
 names=['hidden','conv','state'] if name.startswith('l') and int(name[1:])%4!=3 else ['hidden','past_k','past_v','mask','cos','sin'] if name.startswith('l') and int(name[1:])%4==3 else ['hidden']
 xs=[torch.from_numpy(np.load(src/(n+'.npy'))) for n in names]
 if name.startswith('head'):
  with safe_open(MODEL/'model.safetensors',framework='pt') as f:m=Head(f.get_slice('model.language_model.embed_tokens.weight')[int(name[4:])*32768:min((int(name[4:])+1)*32768,248094)].float(),f.get_tensor('model.language_model.norm.weight').float())
 else:m=(LinearLayer if name.startswith('l') and int(name[1:])%4!=3 else FullLayer)(load_layer(int(name[1:])))
 ys=m(*xs);ys=ys if isinstance(ys,tuple) else (ys,)
 outs=['hidden_out','conv_out','state_out'] if name.startswith('l') and int(name[1:])%4!=3 else ['hidden_out','new_k','new_v'] if name.startswith('l') and int(name[1:])%4==3 else ['logits']
 torch.onnx.export(m,tuple(xs),str(o/'layer0.onnx'),input_names=names,output_names=outs,opset_version=13,dynamo=False)
 for n,x in zip(names,xs):np.save(o/(n+'.npy'),x.numpy())
 (o/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in names)+'\n')
 np.concatenate([x.numpy().ravel() for x in xs]).tofile(o/'input.f32');np.concatenate([y.numpy().ravel() for y in ys]).tofile(o/'reference.f32')
 (o/'shapes.json').write_text(json.dumps({'inputs':{n:list(x.shape) for n,x in zip(names,xs)},'outputs':{n:list(y.shape) for n,y in zip(outs,ys)}},indent=2))
 print('exported',name,flush=True)
