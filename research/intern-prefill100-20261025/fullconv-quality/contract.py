"""Two-variant quality contract, based on immutable fullconv graph contract."""
from pathlib import Path
import copy,collections,hashlib,struct
from base_contract import sha,BINARY_SHA,validate_aba,verify_overlay as verify_aba_overlay
NAMES=['c1280-original','c1280-conv-l2']
def as_aba(cfg):
 assert [x['name'] for x in cfg['variants']]==NAMES
 c=copy.deepcopy(cfg);again=copy.deepcopy(c['variants'][0]);again['name']='c1280-repeat';c['variants'].append(again)
 return c

def validate_quality(cfg):
 assert cfg['semantic_cases']==24 and cfg['natural_max_tokens']==32 and cfg['cost_case_id']=='s64-d64-2'
 assert cfg['repeats']==1 and cfg['quality'] is True and cfg['diagnostic'] is True
 assert len(cfg['feedback_tokens'])==32
 validate_aba(as_aba(cfg));return True

def verify_overlay(cfg):
 validate_quality(cfg);return verify_aba_overlay(as_aba(cfg))

def token_hash(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()
def text_hash(s):return hashlib.sha256(s.encode()).hexdigest()

def validate_fixture(doc,encode,cost_doc,scenes):
 assert len(doc['quality'])==24 and len(scenes)==12 and 'performance' not in doc
 assert doc['systems']==cost_doc['systems']
 source={x['id']:x for x in scenes};assert len(source)==12
 bysystem={s['static_tokens']:s for s in doc['systems']};assert set(bysystem)=={64,128}
 oldbodies={x.get('body',x.get('input')) for kind in ['performance','quality','boundary'] for x in cost_doc.get(kind,[])}
 for s in bysystem.values():
  assert encode(s['prefix'])==s['prefix_tokens'] and token_hash(s['prefix_tokens'])==s['prefix_sha256']
  assert len(encode(s['system_body']))==s['system_body_tokens']==s['static_tokens']-5
 def common(c):
  s=bysystem[c['static_tokens']];st=c['static_tokens'];body=c.get('body',c.get('input'));tokens=encode(c['prompt_text'])
  assert c['tokens']==tokens and c['tokens_sha256']==token_hash(tokens)
  assert c['prefix_tokens']==s['prefix_tokens']==tokens[:st] and token_hash(c['prefix_tokens'])==c['prefix_sha256']
  assert c['body_tokens']==len(encode(body)) and c['template_overhead']==8
  assert c['prompt_tokens']==len(tokens)==st+c['body_tokens']+8
  assert c['expected_reused_tokens']==st and c['expected_new_tokens']==c['body_tokens']+8
  assert c['system_body_tokens']==st-5 and c['system_template_tokens']==5
 for c in doc['quality']:
  common(c);src=source[c['case']];body=c['body'];s=bysystem[c['static_tokens']]
  assert body==c['input']==src['body']==src['background']+src['background_supplement']+src['final'] and body not in oldbodies
  assert c['body_tokens']==src['target'] and c['body_token_ids']==encode(body)
  assert c['body_text_sha256']==text_hash(body) and c['body_tokens_sha256']==token_hash(c['body_token_ids'])
  assert c['prompt_text']==s['prefix']+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n'
  assert c['prompt_text_sha256']==text_hash(c['prompt_text']) and c['dynamic_template_tokens']==8
  assert c['padding']==src['padding'] and c['padding']['strategy']=='none' and not c['padding']['truncated'] and c['padding']['added_token_count']==0
  assert c['gold']['required_action']==src['action'] and c['gold']['forbidden_actions']==src['forbidden']
  assert c['evaluation_mode']=='natural_decode_no_fixed_feedback' and c['cost_fixture'] is False and c['independent_holdout'] is True
 assert len({c['id'] for c in doc['quality']})==24
 counts=collections.Counter((c['static_tokens'],c['body_tokens']) for c in doc['quality'])
 assert set(counts)=={(s,d) for s in [64,128] for d in [64,128,192,256]} and set(counts.values())=={3}
 for key in source:
  a,b=[c for c in doc['quality'] if c['case']==key]
  assert a['body_token_ids']==b['body_token_ids'] and a['gold']==b['gold'] and a['static_tokens']!=b['static_tokens']
 cost=copy.deepcopy(next(c for c in cost_doc['performance'] if c['id']=='s64-d64-2'))
 cost.update(system_body_tokens=59,system_template_tokens=5);common(cost)
 assert cost['static_tokens']==cost['body_tokens']==64 and cost['prompt_tokens']==136
 return cost
