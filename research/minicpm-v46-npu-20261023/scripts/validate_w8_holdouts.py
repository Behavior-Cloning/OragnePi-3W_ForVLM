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
label=sys.argv[1] if len(sys.argv)>1 else 'w8scaled';cases=json.loads((r/'results/holdout-inputs.json').read_text());rows=[json.loads(l) for l in (r/'results'/(label+'-holdouts.jsonl')).read_text().splitlines()];results=[]
for rec in [x for x in rows if x.get('event')=='round']:
 case=cases['cases'][rec['round']];prompt=case['prompt_ids'];ids=rec['ids'];eos=cases['eos_token_id']
 if eos in ids:ids=ids[:ids.index(eos)+1]
 h=hf(torch.tensor([prompt]),use_cache=True);cache=h.past_key_values;checks=[]
 for i,actual in enumerate(ids):
  logits=torch.nn.functional.linear(h.last_hidden_state[:,-1],hf.embed_tokens.weight)[0];pred=int(logits.argmax());checks.append({'step':i,'npu_token':actual,'reference_token_same_history':pred,'match':actual==pred,'npu_token_rank':int((logits>logits[actual]).sum())+1,'logit_gap':float(logits[pred]-logits[actual])})
  h=hf(torch.tensor([[actual]]),past_key_values=cache,use_cache=True);cache=h.past_key_values
 h=hf(torch.tensor([prompt]),use_cache=True);cache=h.past_key_values;ref=[]
 for i in range(16):
  pred=int(torch.nn.functional.linear(h.last_hidden_state[:,-1],hf.embed_tokens.weight).argmax());ref.append(pred)
  if pred==eos:break
  h=hf(torch.tensor([[pred]]),past_key_values=cache,use_cache=True);cache=h.past_key_values
 out={'case':case['case'],'user':case['user'],'text':tok.decode(ids,skip_special_tokens=True),'reference_text':tok.decode(ref,skip_special_tokens=True),'same_history_matches':sum(x['match'] for x in checks),'steps':len(checks),'checks':checks,'prompt_tokens':len(prompt),'ttft_seconds':rec['ttft_seconds'],'decode_tokens_per_second':rec['decode_tokens_per_second']};results.append(out);print(out['user'],out['text'],out['same_history_matches'],'/',out['steps'],'ref:',out['reference_text'],flush=True)
(r/'results'/(label+'-holdouts-quality.json')).write_text(json.dumps({'scope':'four unseen prompts; teacher-forced agreement is numerical smoke evidence, not a general accuracy benchmark','cases':results},ensure_ascii=False,indent=2))
