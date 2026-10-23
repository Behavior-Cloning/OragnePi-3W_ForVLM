from pathlib import Path
import json,numpy as np
from transformers import AutoTokenizer
from model_layers import MODEL,ROOT
r=Path(__file__).resolve().parents[1];tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);out=ROOT/'holdout-prompts';out.mkdir(exist_ok=True);rows=[]
for i,text in enumerate(['我今天工作很累，请温柔地安慰我。','小朋友问二加三等于几，请简单回答。','现在不要移动，请用一句话向我问好。','睡觉之前请给我说两句温柔的晚安。']):
 ids=tok.apply_chat_template([{'role':'system','content':'你是家庭陪伴机器人，简短回答。'},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
 assert len(ids)+16<=128;np.array(ids,np.int32).tofile(out/(str(i)+'.i32'));rows.append({'case':i,'user':text,'prompt_ids':ids,'prompt_tokens':len(ids)})
(r/'results/holdout-inputs.json').write_text(json.dumps({'cases':rows,'eos_token_id':tok.eos_token_id},ensure_ascii=False,indent=2));print(rows,flush=True)
