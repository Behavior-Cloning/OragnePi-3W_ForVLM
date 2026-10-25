from pathlib import Path
import importlib.util,json,hashlib,copy,py_compile
from tokenizers import Tokenizer
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('newbench',D/'board_fullbaseline.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
doc=json.loads((D/'inputs.json').read_text());tok=Tokenizer.from_file(doc['tokenizer_path']);enc=lambda x:tok.encode(x,add_special_tokens=False).ids
cases=m.validate_fixture(doc,enc);assert len(cases)==8
bad=[]
for label,fn in [('wrong-new',lambda d:d['performance'][0].update(expected_new_tokens=64)),('truncated',lambda d:d['performance'][0]['tokens'].pop()),('missing-cell',lambda d:d['performance'].pop()),('wrong-body',lambda d:d['performance'][0].update(body_tokens=63))]:
 t=copy.deepcopy(doc);fn(t)
 try:m.validate_fixture(t,enc)
 except AssertionError:bad.append(label)
 else:raise RuntimeError('badfixture accepted '+label)
for f in D.glob('*.py'):py_compile.compile(str(f),doraise=True)
report={'status':'host_preflight_pass_not_board_run','core_cases':len(doc['performance']),'quality_cases':len(doc['quality']),'diagnostic_core_cells':len(cases),'body_lengths':sorted(set(c['body_tokens'] for c in doc['performance'])),'static_lengths':sorted(set(c['static_tokens'] for c in doc['performance'])),'new_lengths':sorted(set(c['expected_new_tokens'] for c in doc['performance'])),'max_prompt_tokens':max(len(c['tokens']) for c in doc['performance']),'max_KV_with_32_predictions':max(len(c['tokens'])+31 for c in doc['performance']),'rejected_bad_fixtures':bad,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*.py')},'fixture_sha256':hashlib.sha256((D/'inputs.json').read_bytes()).hexdigest()}
(D/'host-preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
