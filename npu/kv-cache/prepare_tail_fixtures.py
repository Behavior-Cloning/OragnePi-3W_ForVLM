"""Freeze natural short turns for a padded-tail prefill experiment, not holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import gguf
from jinja2 import Environment
from full_prompt import tokenizer_from_gguf


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    questions=['Can we chat for a moment?', 'Stay with me for a moment.', 'Tell me something gentle.',
               'How was your day?', 'Please tell me a short story.', 'What can we do together?',
               'Could you keep me company for a little while?', 'Let us talk about a quiet evening.',
               'I had a long day. Could you keep me company?', 'I am feeling tired. Tell me something gentle.',
               'I had a long day. Could you please keep me company?', 'Stay with me.', 'Tell me a story.', "Let's chat."]
    records=[]
    for model,target in [('smol500',20),('intern1b',12)]:
        base=json.loads((root/f'npu/results/aligned-20261013r2/{model}-h512-n32/prompt.json').read_text())
        reader=gguf.GGUFReader(root/base['model_path']);tokenizer=tokenizer_from_gguf(reader)
        template=Environment().from_string(reader.fields['tokenizer.chat_template'].contents())
        history=base['messages'][:-1]
        found=None;lengths=[]
        for question in questions:
            messages=history+[{'role':'user','content':question}]
            payload=messages
            if model=='smol500':payload=[{**m,'content':[{'type':'text','text':m['content']}]} for m in messages]
            text=template.render(messages=payload,add_generation_prompt=True)
            tokens=tokenizer.encode(text,add_special_tokens=False).ids
            assert tokens[:512]==base['token_ids'][:512]
            lengths.append((question,len(tokens)-512))
            if len(tokens)==512+target:
                found=(messages,text,tokens);break
        assert found, (model,'no natural exact-length question',lengths)
        messages,text,tokens=found
        binary=struct.pack('<'+'i'*len(tokens),*tokens)
        case=f'{model}-h512-n{target}'
        folder=a.output/case;folder.mkdir()
        report={**base,'text':'short natural turn for development padded-tail prefill; not independent quality',
                'messages':messages,'rendered_prompt':text,'token_ids':tokens,'new_tokens':target,
                'prompt_sha256':hashlib.sha256(binary).hexdigest()}
        (folder/'prompt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (folder/'prompt.i32').write_bytes(binary)
        records.append({'case':case,'folder':str(folder.resolve()),'history':512,'new_tokens':target,
                        'user_text':messages[-1]['content'],'prompt_sha256':report['prompt_sha256']})
    (a.output/'matrix.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False))


if __name__=='__main__':main()
