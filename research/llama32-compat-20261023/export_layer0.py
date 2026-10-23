from pathlib import Path
import json,hashlib,types
import gguf,numpy as np,torch,onnx,onnxruntime as ort
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_llama3_parameters
from trained_probe import TrainedDecoder,feed_for
from probe import INPUTS,OUTPUTS
r=Path(__file__).resolve().parent;out=Path('/opt/opi-lab/llama32-compat-20261023');out.mkdir(exist_ok=True)
source=r.parent/'vlm-molmo-paligemma-20261023/paligemma/llama-zh';model=source/'FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf'
torch.set_num_threads(2);reader=gguf.GGUFReader(model);tensors={t.name:t for t in reader.tensors}
l=TrainedDecoder(reader,0,stable_norm=True).eval();div=gguf.dequantize(tensors['rope_freqs.weight'].data,tensors['rope_freqs.weight'].tensor_type).copy()
inv=1/(l.base**(np.arange(l.hd//2,dtype=np.float32)/(l.hd//2)))/div
hf,_=_compute_llama3_parameters(LlamaConfig(**json.loads((source/'original-config.json').read_text())),torch.device('cpu'))
np.testing.assert_allclose(inv,hf.numpy(),rtol=3e-6,atol=1e-9)
def rope(self,pos):
 f=pos.float()[:,None]*torch.from_numpy(inv)[None,:];a=torch.cat([f,f],-1)[None,None];return a.cos(),a.sin()
l.rope=types.MethodType(rope,l)
vocab=reader.fields['tokenizer.ggml.tokens'].contents();vmap={x:i for i,x in enumerate(vocab)}
pieces=['Hello',',','ĠI','Ġam','Ġhere','Ġto','Ġhelp','.','Today','Ġis','Ġa','Ġgood','Ġday','.']
ids=[128000]+[vmap[x] for x in pieces];emb=tensors['token_embd.weight'];x=gguf.dequantize(emb.data[ids],emb.tensor_type)[None].copy();cap=32
cache=np.zeros((1,l.kv,cap,l.hd),np.float32);feed=feed_for(l,x[:,:1],cache,cache,0,cap)
path=out/'layer0.onnx';torch.onnx.export(l,tuple(torch.from_numpy(v) for v in feed.values()),str(path),input_names=INPUTS,output_names=OUTPUTS,opset_version=13,dynamo=False)
om=onnx.load(path);om.ir_version=8;onnx.checker.check_model(om,full_check=True);onnx.save(om,path)
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;sess=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
with torch.inference_mode():ref=l.full(torch.from_numpy(x)).numpy()
k=np.zeros_like(cache);v=k.copy();errors=[];data=[]
for pos in range(len(ids)):
 feed=feed_for(l,x[:,pos:pos+1],k,v,pos,cap);z,nk,nv=sess.run(OUTPUTS,feed);k[:,:,pos:pos+1]=nk;v[:,:,pos:pos+1]=nv
 target=ref[:,pos:pos+1];errors.append(dict(pos=pos,max_abs=float(np.max(np.abs(z-target))),relative_l2=float(np.linalg.norm(z-target)/np.linalg.norm(target))))
 np.testing.assert_allclose(z,target,atol=4e-4,rtol=3e-4)
 if pos in [0,1,2,4,8,len(ids)-1]:
  names=[]
  for name,value in feed.items():
   n=f'l0_p{pos}_{name}.npy';np.save(out/n,value);names.append(n)
  data.append(' '.join(names));np.save(out/f'l0_p{pos}_reference.npy',z)
(out/'layer0-dataset.txt').write_text('\n'.join(data)+'\n')
meta=dict(source=str(model),source_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),source_quantization='Q4_K_M GGUF, actual layer0 Q4_K/Q6_K/F32 mixed; not BF16',scope='one real layer0 only, host ONNX check not NPU execution',capacity=cap,dim=l.dim,heads=l.heads,kv=l.kv,head_dim=l.hd,rope_divisors=div.tolist(),rope_hf_max_abs=float(np.max(np.abs(inv-hf.numpy()))),layer_parameter_count=sum(p.numel() for p in l.buffers()),tokens=ids,pieces=pieces,errors=errors,onnx_bytes=path.stat().st_size,onnx_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
(out/'export-summary.json').write_text(json.dumps(meta,indent=2));(r/'export-summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
