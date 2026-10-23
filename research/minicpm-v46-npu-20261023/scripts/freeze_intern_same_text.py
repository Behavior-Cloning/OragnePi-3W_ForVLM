from pathlib import Path
import sys,json
r=Path(__file__).resolve().parents[1];repo=r.parents[1];sys.path.insert(0,str(repo/'npu/kv-cache'))
from resident_http import ChatCodec
model=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
messages=json.loads((r/'results/prompt.json').read_text())['messages'];c=ChatCodec(model,messages[0]['content']);tokens=c.encode(messages[1:]);(r/'results/intern-same-text-input.json').write_text(json.dumps({'messages':messages,'tokens':tokens,'token_count':len(tokens),'tokenizer_source':str(model)},ensure_ascii=False,indent=2));print(len(tokens))
