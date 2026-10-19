"""Export exact tokenizer from the source Q8 GGUF; no model substitution."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'npu/kv-cache'))
import gguf
from full_prompt import tokenizer_from_gguf
p=argparse.ArgumentParser();p.add_argument('gguf',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
r=gguf.GGUFReader(a.gguf);t=tokenizer_from_gguf(r);t.save(str(a.out/'tokenizer.json'))
h=hashlib.sha256()
with a.gguf.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
meta={'template':r.fields['tokenizer.chat_template'].contents(),'source_sha256':h.hexdigest()}
(a.out/'tokenizer-meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(json.dumps({'source_sha256':h.hexdigest(),'tokenizer_sha256':hashlib.sha256((a.out/'tokenizer.json').read_bytes()).hexdigest()}))
