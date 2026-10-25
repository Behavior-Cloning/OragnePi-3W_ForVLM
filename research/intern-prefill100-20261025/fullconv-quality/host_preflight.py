#!/usr/bin/env python3
from pathlib import Path
import copy,json,hashlib,collections,py_compile
from tokenizers import Tokenizer
from contract import validate_quality,validate_fixture,sha
from controller_quality import preflight
P=Path(__file__).resolve().parent
cfg,remote,sources=preflight(P/'config.quality.json','dry-preflight')
doc=json.loads((P/'semantic-inputs.json').read_text());old=json.loads((P/'cost-inputs.json').read_text());scenes=json.loads((P/'scenes.source.json').read_text())
assert doc['tokenizer_sha256']==sha(doc['tokenizer_path'])
assert doc['parent_fixture_sha256']==sha(P/'cost-inputs.json') and doc['fixture_source_sha256']==sha(P/'scenes.source.json')
t=Tokenizer.from_file(doc['tokenizer_path']);enc=lambda s:t.encode(s,add_special_tokens=False).ids
cost=validate_fixture(doc,enc,old,scenes);reject=[]

def bad_config(name,mutation):
 c=copy.deepcopy(cfg);mutation(c)
 try:validate_quality(c)
 except (AssertionError,KeyError,ValueError):reject.append(name)
 else:raise AssertionError('Bad config accepted: '+name)
bad_config('only-short12',lambda c:c.update(semantic_cases=12))
bad_config('wrong-cost-id',lambda c:c.update(cost_case_id='s64-d64-0'))
bad_config('wrong-natural-length',lambda c:c.update(natural_max_tokens=64))
bad_config('dynamic-cache-capacity256',lambda c:c['defaults'].update(capacity=256))
bad_config('different-candidate-flags',lambda c:c['variants'][1]['env'].update(OPI_GROUP_PREFILL='0'))
bad_config('third-variant',lambda c:c['variants'].append(copy.deepcopy(c['variants'][0])))
bad_config('extra-changed-graph',lambda c:c['variants'][1]['expected_graph_sha256'].__setitem__(str(Path(c['variants'][1]['graph'])/'layer0.nb'),'0'*64))
bad_config('wrong-overlay-target',lambda c:c['overlay_contract']['symlinks'].__setitem__('prefill-layer2.nb','/tmp/wrong.nb'))
bad_config('wrong-feedback-length',lambda c:c['feedback_tokens'].pop())
for name,fn in {
 'missing-semantic-case':lambda d:d['quality'].pop(),
 'truncated-token-input':lambda d:d['quality'][0]['tokens'].pop(),
 'wrong-gold':lambda d:d['quality'][0]['gold'].update(required_action='摇头'),
 'wrong-cache-count':lambda d:d['quality'][0].update(expected_reused_tokens=128),
 'wrong-new-count':lambda d:d['quality'][0].update(expected_new_tokens=64),
 'padded-quality':lambda d:d['quality'][0]['padding'].update(added_token_count=1),
 'old-short-in-place-of-long':lambda d:d['quality'].__setitem__(0,copy.deepcopy(old['quality'][0]))}.items():
 d=copy.deepcopy(doc);fn(d)
 try:validate_fixture(d,enc,old,scenes)
 except (AssertionError,KeyError,ValueError):reject.append(name)
 else:raise AssertionError('Bad fixture accepted: '+name)
for f in P.glob('*.py'):py_compile.compile(str(f),doraise=True)
result={'status':'passed_host_only_no_board_access','semantic_cases':24,'independent_bodies':12,'semantic_cells':dict(collections.Counter(f"S{c['static_tokens']}D{c['body_tokens']}" for c in doc['quality'])),
 'cost_case':cost['id'],'cost_prompt_sha256':cost['tokens_sha256'],'cost_prefix_tokens':64,'cost_dynamic_body_tokens':64,'cost_actual_new_tokens':72,
 'counts_total':{'natural_semantic':48,'natural_cost':2,'diagnostic_requests':2,'diagnostic_steps':8,'performance':0},
 'system_total_tokens':[64,128],'system_body_tokens':[59,123],'system_template_tokens':5,'dynamic_template_tokens':8,
 'dynamic_body_tokens':[64,128,192,256],'actual_new_tokens':[72,136,200,264],'natural_output_cap':32,'negative_tests_rejected':reject,
 'fixture_sha256':sha(P/'semantic-inputs.json'),'cost_fixture_sha256':sha(P/'cost-inputs.json'),'scenes_sha256':sha(P/'scenes.source.json'),
 'tokenizer_sha256':sha(doc['tokenizer_path']),'upload_bytes_without_config':sum(f.stat().st_size for f in sources.values()),'board_access':False}
(P/'host-preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
