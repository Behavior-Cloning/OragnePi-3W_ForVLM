from pathlib import Path
import copy,json,hashlib,struct,collections
from tokenizers import Tokenizer
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
hids=lambda a:sha(struct.pack('<'+'i'*len(a),*a))
p=D/'inputs.json';raw=p.read_bytes();j=json.loads(raw);parent=json.loads(Path(j['parent_fixture_path']).read_text())
t=Tokenizer.from_file(j['tokenizer_path']);enc=lambda x:t.encode(x,add_special_tokens=False).ids
assert j['tokenizer_sha256']==sha(Path(j['tokenizer_path']).read_bytes())
assert j['parent_fixture_sha256']==sha(Path(j['parent_fixture_path']).read_bytes())
assert j['systems']==parent['systems']
assert j['fixture_source_sha256']==sha((D/'scenes.source.json').read_bytes())
source={x['id']:x for x in json.loads((D/'scenes.source.json').read_text())}
oldbodies={x.get('body',x.get('input')) for kind in ['performance','quality','boundary'] for x in parent.get(kind,[])}
assert len(j['quality'])==24

def check(c):
 s=next(s for s in j['systems'] if s['static_tokens']==c['static_tokens']);src=source[c['case']]
 body=c['body'];ids=enc(body);allids=enc(c['prompt_text']);st=c['static_tokens'];pfx=enc(s['prefix'])
 assert body==c['input']==src['body'];assert body==src['background']+src['background_supplement']+src['final']
 assert body not in oldbodies
 assert c['body_tokens']==len(ids)==src['target']
 assert c['body_token_ids']==ids and c['body_tokens_sha256']==hids(ids) and c['body_text_sha256']==sha(body.encode())
 assert c['tokens']==allids and c['tokens_sha256']==hids(allids) and c['prompt_text_sha256']==sha(c['prompt_text'].encode())
 assert c['prompt_text']==s['prefix']+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n'
 assert pfx==c['prefix_tokens']==allids[:st] and len(pfx)==st and hids(pfx)==c['prefix_sha256']
 assert c['system_body_tokens']==len(enc(s['system_body'])) and c['system_template_tokens']==5 and st==c['system_body_tokens']+5
 assert c['template_overhead']==c['dynamic_template_tokens']==len(allids)-st-len(ids)==8
 assert c['expected_reused_tokens']==st and c['expected_new_tokens']==len(ids)+8
 assert c['prompt_tokens']==len(allids)==st+c['expected_new_tokens']
 assert c['padding']==src['padding'] and c['padding']['strategy']=='none' and not c['padding']['truncated'] and c['padding']['added_token_count']==0
 assert c['gold']['required_action']==src['action'] and c['gold']['forbidden_actions']==src['forbidden']
 assert c['evaluation_mode']=='natural_decode_no_fixed_feedback'
 assert c['cost_fixture'] is False
for c in j['quality']:check(c)
counts=collections.Counter((c['static_tokens'],c['body_tokens']) for c in j['quality']);assert len(counts)==8 and set(counts.values())=={3}
assert len({c['id'] for c in j['quality']})==24
for case in source:
 a,b=[c for c in j['quality'] if c['case']==case];assert a['body']==b['body'] and a['body_token_ids']==b['body_token_ids'] and a['gold']==b['gold'] and a['static_tokens']!=b['static_tokens']
negative=[]
mutations={'wrong-body-count':lambda c:c.update(body_tokens=c['body_tokens']-1),'wrong-new-count':lambda c:c.update(expected_new_tokens=c['body_tokens']),'truncated-input-ids':lambda c:c['tokens'].pop(),'altered-final-instruction':lambda c:c.update(body=c['body'].replace('点头','摇头')),'wrong-cache-size':lambda c:c.update(expected_reused_tokens=128),'changed-system-prefix':lambda c:c['prefix_tokens'].__setitem__(3,0),'altered-gold':lambda c:c['gold'].update(required_action='摇头')}
for name,fn in mutations.items():
 c=copy.deepcopy(j['quality'][0]);fn(c)
 try:check(c)
 except (AssertionError,ValueError):negative.append(name)
 else:raise RuntimeError('accepted bad fixture '+name)
report={'status':'passed_host_token_validation_not_model_test','cases':24,'independent_scenarios':12,'systems_byte_semantics_equal_parent':j['systems']==parent['systems'],'tokenizer_sha256':j['tokenizer_sha256'],'fixture_sha256':sha(raw),'parent_fixture_sha256':j['parent_fixture_sha256'],'cell_counts':{str(k):v for k,v in sorted(counts.items())},'body_lengths':[64,128,192,256],'actual_new_tokens':[72,136,200,264],'system_body_tokens':[59,123],'system_template_tokens':5,'dynamic_template_tokens':8,'max_prompt_tokens':max(c['prompt_tokens'] for c in j['quality']),'max_KV_if_32_predictions':max(c['prompt_tokens']+31 for c in j['quality']),'dynamic_padding_tokens_total':sum(c['padding']['added_token_count'] for c in j['quality']),'truncated_cases':0,'exact_body_overlap_with_parent_fixtures':0,'natural_response_model_calls':0,'board_access':False,'negative_tests_rejected':negative,'rows':[{'id':c['id'],'static_tokens':c['static_tokens'],'body_tokens':c['body_tokens'],'dynamic_template_tokens':8,'actual_new_tokens':c['expected_new_tokens'],'prompt_tokens':c['prompt_tokens'],'body_text_sha256':c['body_text_sha256'],'tokens_sha256':c['tokens_sha256']} for c in j['quality']],'source_sha256':{x.name:sha(x.read_bytes()) for x in [D/'build_fixture.py',D/'validate_fixture.py',D/'scenes.source.json',D/'GOLD_RUBRIC.md']}}
(D/'token-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_sha256']},ensure_ascii=False,indent=2))
