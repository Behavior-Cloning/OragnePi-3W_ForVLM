from pathlib import Path
import json, hashlib, copy, struct
import tokenizers
from tokenizers import Tokenizer
D=Path(__file__).resolve().parent; R=D.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(R/'inputs.json').read_bytes(); parent=json.loads(raw)
tp=Path(parent['tokenizer_path']); assert sha(tp.read_bytes())==parent['tokenizer_sha256']
tok=Tokenizer.from_file(str(tp)); enc=lambda s:tok.encode(s,add_special_tokens=False).ids
hash_ids=lambda ids:sha(struct.pack('<'+'i'*len(ids),*ids))
# Existing token hash encoding is explicitly checked, rather than guessed.
assert hash_ids(parent['performance'][0]['tokens'])==parent['performance'][0]['tokens_sha256']
seeds=json.loads((D/'scenes.source.json').read_text()); assert len(seeds)==12
body_counts={n:sum(x['target']==n for x in seeds) for n in (64,128,192,256)};assert set(body_counts.values())=={3}
assert len({x['id'] for x in seeds})==12
rows=[]
for s in parent['systems']:
 assert enc(s['prefix'])==s['prefix_tokens']; assert len(s['prefix_tokens'])==s['static_tokens']
 assert len(enc(s['system_body']))==s['system_body_tokens']
 assert s['system_template_tokens']==len(enc(s['prefix']))-len(enc(s['system_body']))==5
 for x in seeds:
  body=x['body']; ids=enc(body); assert len(ids)==x['target'], (x['id'],len(ids))
  assert body==x['background']+x['background_supplement']+x['final']
  assert x['padding']['added_token_count']==0 and not x['padding']['truncated']
  prompt=s['prefix']+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n'
  allids=enc(prompt); static=s['static_tokens']; overhead=len(allids)-static-len(ids); assert overhead==8
  assert allids[:static]==s['prefix_tokens']
  assert allids[static+3:static+3+len(ids)]==ids
  rows.append({'id':'s'+str(static)+'-'+x['id'],'case':x['id'],'static_tokens':static,
  'system_body_tokens':s['system_body_tokens'],'system_template_tokens':s['system_template_tokens'],
  'prefix_tokens':s['prefix_tokens'],'prefix_sha256':s['prefix_sha256'],
  'body':body,'input':body,'body_tokens':len(ids),'body_token_ids':ids,'body_text_sha256':sha(body.encode()),'body_tokens_sha256':hash_ids(ids),
  'template_overhead':overhead,'dynamic_template_tokens':overhead,'prompt_tokens':len(allids),'tokens':allids,'tokens_sha256':hash_ids(allids),
  'prompt_text':prompt,'prompt_text_sha256':sha(prompt.encode()),'expected_reused_tokens':static,'expected_new_tokens':len(ids)+overhead,
  'padding':x['padding'],'scenario':x['id'],'features':x['features'],'source_sections':{'background':x['background'],'background_supplement':x['background_supplement'],'final_instruction':x['final']},
  'gold':{'required_action':x['action'],'forbidden_actions':x['forbidden'],'judgment':'manual_semantic_not_string_match','static_hold_class':'pass' if x['action']=='保持全身静止' else 'safe_underresponse','fullbody_immobility_required':x['action']=='保持全身静止'},
  'task_constraint_for_manual_audit':x['action']+'；禁止：'+'、'.join(x['forbidden']),
  'evaluation_mode':'natural_decode_no_fixed_feedback','independent_holdout':True,'holdout_definition':'newly authored before any model execution; not a training-data-disjoint claim','quality_use':'semantic_long_holdout','cost_fixture':False})
assert len(rows)==24
out={'schema_version':1,'name':'semantic-long-manual-holdout-v1','created':'2026-10-25','status':'fixtures_frozen_not_model_tested','scope':'text-only Chinese household/desktop companion action planning; mock hardware state is supplied in text',
 'parent_fixture_path':str(R/'inputs.json'),'parent_fixture_sha256':sha(raw),'systems':copy.deepcopy(parent['systems']),
 'tokenizer_path':str(tp),'tokenizer_sha256':sha(tp.read_bytes()),'tokenizers_version':tokenizers.__version__,'add_special_tokens':False,'token_hash_encoding':parent['token_hash_encoding'],
 'encoding_baseline':parent['encoding_baseline'],'definitions':{'static_tokens':'entire cached system prefix, includes 5 system template tokens; exact parent systems retained','body_tokens':'user scenario only, real Intern tokenizer; 64/128/192/256 exactly','dynamic_template_tokens':'8 user/assistant template tokens outside body','expected_new_tokens':'body_tokens + 8; no precomputed or reused dynamic KV','quality':'natural autoregressive response, no fixed feedback; not wired to a harness','independent_scenarios':12,'system_presentations':24,'padding':'zero dynamic padding; inherited system128 repetition/punctuation deliberately unchanged','comparison_limit':'same cases permit paired candidate/baseline comparisons; distinct semantics across lengths do not isolate a causal length effect'},
 'robot_capabilities':{'head':True,'two_arms':True,'waist':True,'base':'low-DOF front/back/left/right translation and turning','fingers':False,'eyes':False,'screen':False,'display_expression':False,'object_grasping_allowed':False,'real_motor_execution':False},
 'quality':rows,'expected_counts':{'body_lengths':[64,128,192,256],'systems':[64,128],'scenarios_per_body_length':3,'unique_scenarios':12,'presentations':24,'new_token_lengths':[72,136,200,264]},
 'split_policy':{'source':'manual new scenarios; no model outputs inspected during authoring','same_body_across_systems':True,'same_system_as_parent':True,'not_training_disjoint_certified':True,'not_independent_24_samples':True,'task_coverage':'negation, correction, source attribution, posture/state dependency, clearance, direction and degree limits','semantic_freeze':'Do not adjust gold or texts after seeing A/B outputs; revisions require a new fixture version.'},
 'fixture_source_sha256':sha((D/'scenes.source.json').read_bytes())}
(D/'inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'presentations':len(rows),'scenarios':len(seeds),'body_counts':body_counts,'new_token_counts':sorted(set(x['expected_new_tokens'] for x in rows)),'max_prompt':max(x['prompt_tokens'] for x in rows),'fixture_sha256':sha((D/'inputs.json').read_bytes())},indent=2))
