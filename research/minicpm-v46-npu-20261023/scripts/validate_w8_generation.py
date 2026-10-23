from pathlib import Path
import torch,json,sys,time
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel,Qwen3_5TextRotaryEmbedding
from model_layers import MODEL,CFG
r=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.set_grad_enabled(False)
with safe_open(MODEL/'model.safetensors',framework='pt') as f:state={k[len('model.language_model.'):]:f.get_tensor(k).float() for k in f.keys() if k.startswith('model.language_model.')}
with torch.device('meta'):hf=Qwen3_5TextModel(CFG)
hf.load_state_dict(state,assign=True,strict=True);del state;hf.rotary_emb=Qwen3_5TextRotaryEmbedding(CFG);hf.eval();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);prompt=json.loads((r/'results/prompt.json').read_text())['ids']
for label in sys.argv[1:]:
 rows=[json.loads(l) for l in (r/'results'/(label+'-runtime.jsonl')).read_text().splitlines()];rec=next(x for x in rows if x.get('event')=='round');ids=rec['ids'];h=hf(torch.tensor([prompt]),use_cache=True);cache=h.past_key_values;checks=[]
 for i,actual in enumerate(ids):
  logits=torch.nn.functional.linear(h.last_hidden_state[:,-1],hf.embed_tokens.weight)[0];pred=int(logits.argmax());checks.append({'step':i,'npu_token':actual,'reference_token_same_history':pred,'match':actual==pred,'npu_token_rank':int((logits>logits[actual]).sum())+1,'logit_gap':float(logits[pred]-logits[actual])})
  h=hf(torch.tensor([[actual]]),past_key_values=cache,use_cache=True);cache=h.past_key_values
 out={'label':label,'prompt':tok.decode(prompt),'text':tok.decode(ids),'tokens':ids,'original_fp32_teacher_forced_matches':sum(x['match'] for x in checks),'steps':len(checks),'checks':checks,'scope':'same-history next-token agreement against original BF16-derived FP32 reference; single calibration greeting, not held-out task accuracy'}
 (r/'results'/(label+'-quality.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2));print(label,out['text'],out['original_fp32_teacher_forced_matches'],'/',len(checks),flush=True)
