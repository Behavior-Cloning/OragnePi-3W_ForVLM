from pathlib import Path
import sys,json,torch,numpy as np,hashlib,subprocess,time
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from model_layers import *
torch.set_num_threads(4);torch.set_grad_enabled(False)
r=Path(__file__).resolve().parents[1];b=ROOT/'full';b.mkdir(exist_ok=True);cfg=CFG
with safe_open(MODEL/'model.safetensors',framework='pt') as f:
 emb=f.get_tensor('model.language_model.embed_tokens.weight').float();norm=f.get_tensor('model.language_model.norm.weight').float()
 np.asarray(emb,dtype=np.float16).tofile(b/'embeddings.f16')
 state={k[len('model.language_model.'):]:f.get_tensor(k).float() for k in f.keys() if k.startswith('model.language_model.')}
with torch.device('meta'):hf=Qwen3_5TextModel(cfg)
hf.load_state_dict(state,assign=True,strict=True);del state;hf.eval()
mods=[LinearLayer(l) if l.layer_type=='linear_attention' else FullLayer(l) for l in hf.layers]
tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
messages=[{'role':'system','content':'你是家庭陪伴机器人，简短回答。'},{'role':'user','content':'你好，请介绍一下自己。'}]
prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
assert len(prompt)+16<=128;np.array(prompt,np.int32).tofile(b/'prompt.i32');(r/'results/prompt.json').write_text(json.dumps({'messages':messages,'ids':prompt,'count':len(prompt),'thinking':False},ensure_ascii=False,indent=2))
caches=[]
for i,m in enumerate(mods):
 if i%4!=3:caches.append([torch.zeros(1,6144,4),torch.zeros(1,16,128,128)])
 else:caches.append([torch.zeros(1,2,127,256),torch.zeros(1,2,127,256)])
manifest=[];custom_tokens=[];cache=None;maxdiff=0.
for pos,token in enumerate(prompt):
 h=emb[token].reshape(1,1,1024)
 angles=pos/(10000000**(torch.arange(0,64,2).float()/64));angles=torch.cat([angles,angles]);cos=angles.cos().reshape(1,1,1,64);sin=angles.sin().reshape(1,1,1,64)
 mask=torch.full((1,1,1,128),-10000.);mask[:,:,:,-(pos+1):]=0
 for i,m in enumerate(mods):
  xs=(h,*caches[i]) if i%4!=3 else (h,*caches[i],mask,cos,sin)
  ys=m(*xs)
  if pos==len(prompt)-1:
   folder=b/('l'+str(i));folder.mkdir(exist_ok=True);names=['hidden','conv','state'] if i%4!=3 else ['hidden','past_k','past_v','mask','cos','sin']
   torch.onnx.export(m,xs,str(folder/'layer0.onnx'),input_names=names,output_names=['hidden_out','conv_out','state_out'] if i%4!=3 else ['hidden_out','new_k','new_v'],opset_version=13,dynamo=False)
   for name,x in zip(names,xs):np.save(folder/(name+'.npy'),x.numpy())
   (folder/'layer0-dataset.txt').write_text(' '.join(n+'.npy' for n in names)+'\n');np.concatenate([x.numpy().ravel() for x in xs]).tofile(folder/'input.f32');np.concatenate([x.numpy().ravel() for x in ys]).tofile(folder/'reference.f32')
   manifest.append({'layer':i,'type':cfg.layer_types[i],'input_counts':[x.numel() for x in xs],'output_counts':[y.numel() for y in ys]});print('export',i,flush=True)
  h=ys[0]
  if i%4!=3:caches[i]=list(ys[1:])
  else:caches[i]=[torch.cat([old[:,:,1:,:],new],dim=2) for old,new in zip(caches[i],ys[1:])]
 ref=hf(torch.tensor([[token]]),past_key_values=cache,use_cache=True);cache=ref.past_key_values
 normalized=h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-6)*(1+norm)
 err=float((normalized-ref.last_hidden_state).abs().max());maxdiff=max(maxdiff,err)
 if err>.02:raise RuntimeError('HF reference mismatch '+str((pos,err)))
logits=torch.nn.functional.linear(normalized,emb);first=int(logits[0,0].argmax());ref_first=int(torch.nn.functional.linear(ref.last_hidden_state,emb)[0,0].argmax());assert first==ref_first
ref_ids=[first]
for _ in range(15):
 out=hf(torch.tensor([[ref_ids[-1]]]),past_key_values=cache,use_cache=True);cache=out.past_key_values;ref_ids.append(int(torch.nn.functional.linear(out.last_hidden_state,emb)[0,0].argmax()))
for i,start in enumerate(range(0,len(emb),32768)):
 folder=b/('head'+str(i));folder.mkdir(exist_ok=True);m=Head(emb[start:start+32768].contiguous(),norm);torch.onnx.export(m,(h,),str(folder/'layer0.onnx'),input_names=['hidden'],output_names=['logits'],opset_version=13,dynamo=False);np.save(folder/'hidden.npy',h.numpy());(folder/'layer0-dataset.txt').write_text('hidden.npy\n');h.numpy().tofile(folder/'input.f32');m(h).numpy().tofile(folder/'reference.f32');print('head',i,flush=True)
(r/'results/export-full.json').write_text(json.dumps({'layers':manifest,'prompt_tokens':len(prompt),'reference_max_abs':maxdiff,'first_token':first,'reference_ids':ref_ids,'reference_text':tokenizer.decode(ref_ids),'head_shards':8,'head_chunk':32768,'vocab':248094,'capacity':128,'precision':'FP32 source; target FP16 preliminary'},indent=2,ensure_ascii=False));print('EXPORT_COMPLETE',flush=True)
