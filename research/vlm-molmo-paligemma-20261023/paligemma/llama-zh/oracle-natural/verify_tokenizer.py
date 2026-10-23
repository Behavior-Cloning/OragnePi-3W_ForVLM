import pathlib,sys,json,hashlib
P=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(pathlib.Path(__file__).parent/'source-snapshot'))
from model import *
from tokenizers import Tokenizer
reader,_=load();rebuilt=build_tokenizer(reader);original=Tokenizer.from_file(str(P/'original-tokenizer.json'))
spec=json.loads((BUILD/'inputs.json').read_text())
texts=['你好，我今天有点累。','明早七点提醒我起床。','不要开灯，请陪我聊两句。','12、1234、2026年10月23日','hello\nworld',spec['prefix_text']]+[c['prompt'] for c in spec['cases']]
rows=[]
for text in texts:
 a=rebuilt.encode(text,add_special_tokens=False);b=original.encode(text,add_special_tokens=False).ids
 rows.append(dict(text=text,gguf_ids=a,original_ids=b,match=a==b))
assert all(r['match'] for r in rows),rows
for c in spec['cases']:assert original.encode(c['prompt'],add_special_tokens=False).ids==c['tokens']
res=dict(source_tokenizer_sha256=sha(P/'original-tokenizer.json'),model=str(MODEL),rows=rows,all_match=True,frozen_prompt_tokens=[len(c['tokens']) for c in spec['cases']])
(pathlib.Path(__file__).parent/'tokenizer-parity.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(dict(all_match=True,cases=len(rows),frozen_prompt_tokens=res['frozen_prompt_tokens'])))
