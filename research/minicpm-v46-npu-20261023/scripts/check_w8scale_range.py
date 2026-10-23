from pathlib import Path
import torch,numpy as np,json,time,hashlib
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel,Qwen3_5TextRotaryEmbedding
from model_layers import MODEL,CFG,ROOT
r=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.time()
with safe_open(MODEL/'model.safetensors',framework='pt') as f:state={k[len('model.language_model.'):]:f.get_tensor(k).float() for k in f.keys() if k.startswith('model.language_model.')}
with torch.device('meta'):hf=Qwen3_5TextModel(CFG)
hf.load_state_dict(state,assign=True,strict=True);del state;hf.rotary_emb=Qwen3_5TextRotaryEmbedding(CFG);hf.eval();ranges={};hashes={}
def update(key,x,y):
 v=ranges.setdefault(key,{'input_absmax':0.,'output_absmax':0.,'rows':0})
 v['input_absmax']=max(v['input_absmax'],float(x.abs().max()));v['output_absmax']=max(v['output_absmax'],float(y.abs().max()));v['rows']+=x.numel()//x.shape[-1]
for name,m in hf.named_modules():
 if isinstance(m,torch.nn.Linear):
  hashes[name]=hashlib.sha256(m.weight.detach().numpy().T.copy().tobytes()).hexdigest()
  m.register_forward_hook(lambda module,args,output,key=name:update(key,args[0],output))
core_peaks={}
for i,l in enumerate(hf.layers):
 if l.layer_type=='linear_attention':
  l.linear_attn.norm.register_forward_pre_hook(lambda m,a,i=i:core_peaks.update({str(i):max(core_peaks.get(str(i),0.),float(a[0].abs().max()))}))
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
texts=['你好，请介绍一下自己。','早上好，今天陪我聊一会儿。','我有一点难过，你能安慰我吗？','请向左转一点，然后停下来。','不要向前移动，只点一下头。','给我讲一个适合孩子听的小故事。','三加五等于多少？请简短回答。','我喜欢安静，晚上九点以后小声说话。']
records=[]
for case,text in enumerate(texts):
 ids=tok.apply_chat_template([{'role':'system','content':'你是家庭陪伴机器人，简短回答。'},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
 cache=None;h=hf(torch.tensor([ids]),use_cache=True);cache=h.past_key_values
 # Calibrate tied output head on all prompt positions, and four valid generated states.
 pred=[]
 for j in range(5):
  hs=h.last_hidden_state
  all_logits=[]
  for i,a in enumerate(range(0,248094,32768)):
   w=hf.embed_tokens.weight[a:a+32768];y=torch.nn.functional.linear(hs,w);update('head'+str(i),hs,y);all_logits.append(y[:,-1,:])
  token=int(torch.cat(all_logits,dim=-1).argmax());pred.append(token)
  if j<4:h=hf(torch.tensor([[token]]),past_key_values=cache,use_cache=True);cache=h.past_key_values
 records.append({'id':case,'user':text,'prompt_ids':ids,'calibration_generated_ids':pred});print('case',case,'tokens',len(ids),flush=True)
result={'scope':'activation calibration only, not quality evaluation','texts':records,'ranges':ranges,'gated_norm_input_absmax':core_peaks,'fixed_gain32_square_overflow_bound':(65504**.5)/32,'weight_sha256':hashes,'seconds':time.time()-start,'source':'original BF16 weights converted to FP32 official transformers 5.7.0 CPU','headroom_bits':1}
(r/'results/w8scale-range-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('COMPLETE',time.time()-start,flush=True)
