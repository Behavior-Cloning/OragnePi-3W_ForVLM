import json,collections
from pathlib import Path
from transformers import AutoTokenizer
r=Path(__file__).resolve().parent;actions=json.loads((r/'actions.json').read_text());results={}
for name,path in [('nano33','models/nano33'),('laya','models/laya/tokenizer'),('jev08','models/jev08/tokenizer')]:
 tok=AutoTokenizer.from_pretrained(r/path,local_files_only=True);groups=collections.defaultdict(list);ids=[]
 for a in actions:
  z=tok.encode(a['description'],add_special_tokens=False);groups[tuple(z)].append(a['id']);ids.extend(z)
 results[name]={'unique_candidate_token_sequences':len(groups),'candidate_count':len(actions),'unk_fraction':sum(x==tok.unk_token_id for x in ids)/len(ids),'collisions':[v for v in groups.values() if len(v)>1]}
(r/'results/chinese-tokenizer-audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False,indent=2))
