from pathlib import Path
import json
from transformers import AutoTokenizer
r=Path(__file__).resolve().parents[1];tok=AutoTokenizer.from_pretrained('/mnt/f/Files/OPi-Lab/models/MiniCPM-V-4.6',local_files_only=True)
rows=[]
for line in (r/'results/baseline-runtime.jsonl').read_text().splitlines():
 j=json.loads(line)
 if 'ids' in j:j['text']=tok.decode(j['ids'])
 rows.append(j)
(r/'results/baseline-decoded.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False));print([(v.get('event'),v.get('text')) for v in rows])
