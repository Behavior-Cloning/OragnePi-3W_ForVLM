from pathlib import Path
import json,hashlib,struct,time
from tokenizers import Tokenizer
D=Path(__file__).resolve().parent;R=D.parent
old=json.loads((R/'fullconv-quality/semantic-inputs.json').read_text());src=json.loads((D/'scenes.source.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(old['tokenizer_path']))==old['tokenizer_sha256'];tok=Tokenizer.from_file(old['tokenizer_path']);enc=lambda s:tok.encode(s,add_special_tokens=False).ids
h=lambda x:hashlib.sha256(struct.pack('<'+str(len(x))+'I',*x)).hexdigest()
quality=[]
for scene in src['scenes']:
 core=scene['core'];target=scene['body_tokens'];pool=src['background_sentence_pool'];states={len(enc(core)):('',[])}
 assert len(enc(core))<=target,(scene['id'],len(enc(core)),target)
 for idx,sentence in enumerate(pool):
  for length,(text,ids) in list(states.items()):
   text2=text+sentence;n=len(enc(text2+core))
   if n<=target and n not in states:states[n]=(text2,ids+[idx])
  if target in states:break
 assert target in states,(scene['id'],sorted(states))
 supplement,used=states[target];body=supplement+core;bodyids=enc(body);assert len(bodyids)==target
 for system in old['systems']:
  prefix=system['prefix'];prefixids=system['prefix_tokens'];static=len(prefixids);prompt=prefix+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n';ids=enc(prompt)
  assert ids[:static]==prefixids and len(ids)==static+target+8
  quality.append({'id':f"s{static}-d{target}-{scene['id']}",'case':scene['id'],'scenario':scene['id'],'static_tokens':static,'system_body_tokens':system['system_body_tokens'],'system_template_tokens':5,'prefix_tokens':prefixids,'prefix_sha256':h(prefixids),'body':body,'input':body,'body_tokens':target,'body_token_ids':bodyids,'body_tokens_sha256':h(bodyids),'template_overhead':8,'dynamic_template_tokens':8,'prompt_tokens':len(ids),'tokens':ids,'tokens_sha256':h(ids),'prompt_text':prompt,'expected_reused_tokens':static,'expected_new_tokens':target+8,'gold':{'required_action':scene['required'],'forbidden_actions':scene['forbidden'],'judgment':'manual semantic compatibility, full constraint completion separate'},'background_composition':{'sentence_indices':used,'complete_sentences':True,'truncated':False,'purpose':'fixed body token count; artificial background composition, not natural user speech corpus'},'model_outputs_used_to_select':False})
assert len(quality)==24
out={'name':'independent-household-v2-frozen','status':'not_model_tested','created_unix':time.time(),'systems':old['systems'],'tokenizer_path':old['tokenizer_path'],'tokenizer_sha256':old['tokenizer_sha256'],'quality':quality,'source_sha256':sha(D/'scenes.source.json'),'generator_sha256':sha(Path(__file__)),'scope':'12 new household action scenarios,2 fixed systems, exact body64/128/192/256; complete background composition; no output-based tuning, no training-disjoint claim','expected_counts':{'unique_scenarios':12,'presentations':24},'calibration_use':'forbidden before first frozen-candidate evaluation'}
(D/'semantic-inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print([(x['id'],x['body_tokens'],x['prompt_tokens']) for x in quality]);print('sha256',sha(D/'semantic-inputs.json'))
