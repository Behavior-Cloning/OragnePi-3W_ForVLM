from pathlib import Path
import json,numpy as np,torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_llama3_parameters
from model import *
torch.set_num_threads(2);BUILD.mkdir(parents=True,exist_ok=True);reader,tensors=load();tok=build_tokenizer(reader)
texts=['你好，我今天有点累。','明早七点提醒我起床。','不要开灯，请陪我聊两句。','12、1234、2026年10月23日','hello\nworld']
for t in texts:assert tok.decode(tok.encode(t,add_special_tokens=False),skip_special_tokens=False)==t
system='你是家庭陪伴机器人，请用简短中文回答。'
prefix_text=tok.apply_chat_template([dict(role='system',content=system)],tokenize=False,add_generation_prompt=False)
prefix=tok.encode(prefix_text,add_special_tokens=False);cases=[]
for i,text in enumerate(texts[:3]):
 prompt=tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=text)],tokenize=False,add_generation_prompt=True);ids=tok.encode(prompt,add_special_tokens=False);assert ids[:len(prefix)]==prefix
 cases.append(dict(id='zh'+str(i),text=text,prompt=prompt,tokens=ids,dynamic_tokens=len(ids)-len(prefix)))
inv=inverse_freq(reader);cfg=LlamaConfig(**json.loads((SOURCE/'original-config.json').read_text()));hf,_=_compute_llama3_parameters(cfg,torch.device('cpu'));np.testing.assert_allclose(inv,hf.numpy(),rtol=3e-6,atol=1e-9)
inv.astype('<f4').tofile(BUILD/'rope_inv.f32');(BUILD/'tokenizer').mkdir(exist_ok=True);tok.save_pretrained(BUILD/'tokenizer')
inputs=dict(source=str(MODEL),source_sha256=sha(MODEL),source_precision=MODEL.name+' dequantized reference; not original BF16',system=system,prefix_text=prefix_text,prefix=prefix,cases=cases,capacity=256,max_new=32,eos=list(cfg.eos_token_id),tokenizer='GGUF vocab+merges with Llama3 split regex; CPU runtime tokenization parity still required',rope_max_abs=float(np.max(np.abs(inv-hf.numpy()))))
(ROOT/('inputs-'+SOURCE_TAG+'.json')).write_text(json.dumps(inputs,ensure_ascii=False,indent=2));(BUILD/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2));print(json.dumps(dict(prefix_tokens=len(prefix),case_tokens=[len(x['tokens']) for x in cases],rope_max_abs=inputs['rope_max_abs']),ensure_ascii=False))

(ROOT/'llama_eos.h').write_text('static inline int llama_is_eos(int token,int configured){return token==configured'+''.join('||token=='+str(x) for x in cfg.eos_token_id)+';}\n')
