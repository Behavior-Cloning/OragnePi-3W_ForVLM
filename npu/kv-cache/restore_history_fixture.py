"""Reconstruct an archived CPU continuation and verify the original prompt hash."""
import argparse
import hashlib
import json
from pathlib import Path
import gguf
import numpy as np
from jinja2 import Environment
from full_prompt import tokenizer_from_gguf

p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
p.add_argument('--model',type=Path,required=True);p.add_argument('--model-id',required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.archive.read_text());m=next(m for m in r['models'] if m['id']==a.model_id)
c=next(c for c in m['contexts'] if c['context']==1024)['cases'][0]
reader=gguf.GGUFReader(a.model);tokenizer=tokenizer_from_gguf(reader)
template=Environment().from_string(reader.fields['tokenizer.chat_template'].contents())
def render(messages):
    if reader.fields['tokenizer.ggml.pre'].contents()=='smollm':
        messages=[{**m,'content':[{'type':'text','text':m['content']}]} for m in messages]
    text=template.render(messages=messages,add_generation_prompt=True)
    return text,tokenizer.encode(text,add_special_tokens=False).ids
initial,tokens=render(c['messages'])
assert hashlib.sha256(initial.encode()).hexdigest()==c['prompt_sha256']
assert tokens==c['prompt_tokens']
text,tokens=render(c['incremental']['messages']);timing=c['incremental']['run']['timings']
assert len(tokens)==c['incremental']['input_tokens']==timing['cache_n']+timing['prompt_n']
a.output.mkdir(parents=True,exist_ok=False)
report={'text':'archived short-history companion continuation','rendered_prompt':text,'token_ids':tokens,
        'messages':c['incremental']['messages'],'archive':str(a.archive),
        'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),
        'cached_prefix':timing['cache_n'],'new_tokens':timing['prompt_n'],
        'historical_cpu_timing':timing,'historical_cpu_client_ttft_seconds':c['incremental']['run']['ttft_seconds'],
        'scope':'same archived input; new generation must be measured, not inferred from old speed'}
(a.output/'prompt.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
np.asarray(tokens,dtype='<i4').tofile(a.output/'prompt.i32')
print(json.dumps({k:v for k,v in report.items() if k not in ('rendered_prompt','token_ids','messages')},ensure_ascii=False,indent=2))
