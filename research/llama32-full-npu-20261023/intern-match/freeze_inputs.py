from pathlib import Path
from tokenizers import Tokenizer
import json,hashlib
R=Path(__file__).resolve().parent;llama=json.loads((R.parent/'inputs.json').read_text());path=Path('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json');t=Tokenizer.from_file(str(path));enc=lambda s:t.encode(s,add_special_tokens=False).ids
system=llama['system'];prefix_text='<|im_start|>system\n'+system+'<|im_end|>\n';prefix=enc(prefix_text);cases=[]
for c in llama['cases']:
 prompt=prefix_text+'<|im_start|>user\n'+c['text']+'<|im_end|>\n<|im_start|>assistant\n';ids=enc(prompt);assert ids[:len(prefix)]==prefix;cases.append(dict(id=c['id'],text=c['text'],prompt=prompt,tokens=ids,dynamic_tokens=len(ids)-len(prefix)))
assert len(prefix)==16 and [len(x['tokens']) for x in cases]==[31,32,33]
spec=dict(source='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018',source_sha256='Graph SHA captured on board; historical Q8 source',source_precision='Existing Intern W8 FC graphs with layer2/3 down FP16 exceptions; all four heads W8',system=system,prefix_text=prefix_text,prefix=prefix,cases=cases,capacity=1280,max_new=96,eos=[151645],tokenizer_path=str(path),tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cache_policy='only system16, hard cap',prefill='batch16 with tail_min8; differs from Llama scalar prefill')
(R/'inputs.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2));print(json.dumps(dict(prefix=len(prefix),prompts=[len(x['tokens']) for x in cases])))
