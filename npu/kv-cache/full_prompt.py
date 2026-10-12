"""Create a text-only SmolVLM chat prompt from its embedded GGUF tokenizer."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
from jinja2 import Environment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

def tokenizer_from_gguf(reader):
    assert reader.fields['tokenizer.ggml.pre'].contents()=='smollm'
    vocab=reader.fields['tokenizer.ggml.tokens'].contents()
    merges=[tuple(v.split(' ',1)) for v in reader.fields['tokenizer.ggml.merges'].contents()]
    tokenizer=Tokenizer(models.BPE(vocab={v:i for i,v in enumerate(vocab)},merges=merges))
    tokenizer.pre_tokenizer=pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r'\p{N}'),behavior='isolated'),
        pre_tokenizers.Split(Regex(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"),behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
    tokenizer.decoder=decoders.ByteLevel()
    types=reader.fields['tokenizer.ggml.token_type'].contents()
    tokenizer.add_special_tokens([AddedToken(v,normalized=False,special=True) for v,t in zip(vocab,types) if t in (3,4)])
    return tokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--text',default='Hello! Please introduce yourself in one short sentence.')
    args=p.parse_args();reader=gguf.GGUFReader(args.model);tokenizer=tokenizer_from_gguf(reader)
    messages=[{'role':'user','content':[{'type':'text','text':args.text}]}]
    template=reader.fields['tokenizer.chat_template'].contents()
    prompt=Environment().from_string(template).render(messages=messages,add_generation_prompt=True)
    encoded=tokenizer.encode(prompt,add_special_tokens=False)
    args.output.mkdir(parents=True,exist_ok=True)
    tokenizer.save(str(args.output/'tokenizer.json'))
    np.asarray(encoded.ids,dtype=np.int32).tofile(args.output/'prompt.i32')
    (args.output/'prompt.json').write_text(json.dumps({'text':args.text,'rendered_prompt':prompt,
        'token_ids':encoded.ids,'pieces':encoded.tokens,'tokenizer_verification':'pending independent llama.cpp check'},indent=2,ensure_ascii=False)+'\n')
    print(prompt);print(encoded.ids)

if __name__=='__main__':main()
