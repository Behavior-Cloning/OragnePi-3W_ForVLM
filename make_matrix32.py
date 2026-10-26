#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,struct
from tokenizers import Tokenizer
T=Path(__file__).resolve().parent/'research/intern-prefill100-20261025'
base=json.loads((T/'inputs.json').read_text())
tok=Tokenizer.from_file(base['tokenizer_path']); enc=lambda s:tok.encode(s,add_special_tokens=False).ids
shaids=lambda ids:hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()
shabytes=lambda b:hashlib.sha256(b).hexdigest()
# Build a complete 32-token system prefix: 3-token start + 27-token body + 2-token end.
old=base['systems'][0]; start='<|im_start|>system\n'; end='<|im_end|>\n'
old_body_ids=enc(old['system_body'])
assert len(old_body_ids)==old['system_body_tokens']==59
body32=tok.decode(old_body_ids[:27],skip_special_tokens=False)
prefix32=start+body32+end
ids32=enc(prefix32)
assert len(ids32)==32 and ids32==enc(prefix32)
system32={'static_tokens':32,'system_body_tokens':27,'system_template_tokens':5,'system_body':body32,'prefix':prefix32,'prefix_tokens':ids32,'prefix_sha256':shaids(ids32),'cost_fixture':True,'note':'27-token truncated fixed system body; performance cost fixture, not semantic quality sample'}
# Preserve full 64 system as semantic reference.
system64=next(s for s in base['systems'] if s['static_tokens']==64)
systems=[system32,system64]
by={(c['static_tokens'],c['body_tokens']):c for c in base['performance']}
# Dynamic body 32 is a token-exact prefix of the first 64-token body; other lengths use existing frozen text.
seed=by[(64,64)]['body']; seed_ids=enc(seed); assert len(seed_ids)==64
body32txt=tok.decode(seed_ids[:32],skip_special_tokens=False); assert len(enc(body32txt))==32
out=[]
for static,system in [(32,system32),(64,system64)]:
 for d in [32,64,128,256]:
  for rep in range(3):
   if d==32: body=body32txt; src=by[(64,64)]
   else: body=by[(64,d)]['body']; src=by[(64,d)]
   assert len(enc(body))==d
   # User/assistant suffix is taken from the canonical static64 case after its prefix.
   _u='<|im_start|>user\n'; _bs=src['prompt_text'].index(_u)+len(_u); suffix=src['prompt_text'][_bs+len(src['body']):]
   prompt=system['prefix']+_u+body+suffix
   ids=enc(prompt)
   assert ids[:static]==system['prefix_tokens'] and len(ids)==static+d+8,(static,d,len(ids))
   out.append({'id':f's{static}-d{d}-{rep}','static_tokens':static,'system_body_tokens':system['system_body_tokens'],'system_template_tokens':5,'prefix_tokens':system['prefix_tokens'],'prefix_sha256':system['prefix_sha256'],'scenario':src.get('scenario',f'cost-s{static}-d{d}'),'body':body,'body_tokens':d,'body_sha256':shaids(enc(body)),'padding':{'cost_fixture':True,'source_case':src['id'],'truncated_body':d==32},'template_overhead':8,'tokens':ids,'prompt_tokens':len(ids),'prompt_text':prompt,'tokens_sha256':shaids(ids),'prompt_text_sha256':shabytes(prompt.encode()),'expected_reused_tokens':static,'expected_new_tokens':d+8,'timing':'32 fixed-feedback output steps; TTFT ends at first token','quality_use':False})
assert len(out)==24 and {(c['static_tokens'],c['body_tokens']) for c in out}=={(s,d) for s in [32,64] for d in [32,64,128,256]}
result={'schema_version':3,'date':'2026-10-26','tokenizer_path':base['tokenizer_path'],'encoding_baseline':base['encoding_baseline'],'tokenizer_sha256':base['tokenizer_sha256'],'tokenizers_version':base['tokenizers_version'],'add_special_tokens':False,'token_hash_encoding':base['token_hash_encoding'],'generator_sha256':shabytes(Path(__file__).read_bytes()),'definitions':{'system_tokens':'static32/64; 32 is cost-only truncated fixed body','dynamic_tokens':'32/64/128/256 body + 8 role/template tokens','cache_policy':'Only frozen system KV; dynamic KV reuse forbidden'},'systems':systems,'performance':out,'quality':[],'scope':'fair cost matrix static32/64 x dynamic32/64/128/256; performance only; no semantic quality claim','parent_fixture_sha256':shabytes((T/'inputs.json').read_bytes())}
(T/'matrix32-inputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(T/'matrix32-inputs.json'),'sha256':shabytes((T/'matrix32-inputs.json').read_bytes()),'systems':[(s['static_tokens'],s['system_body_tokens']) for s in systems],'cells':len(out),'rows':[(s,d,d+8,s+d+8) for s in [32,64] for d in [32,64,128,256]]},ensure_ascii=False))
