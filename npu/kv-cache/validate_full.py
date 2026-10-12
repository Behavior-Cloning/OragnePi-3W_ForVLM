"""Check NPU logits against full-depth FP32 for the NPU's actual token trajectory."""
import argparse
import json
from pathlib import Path
import gguf
import numpy as np
import torch
from tokenizers import Tokenizer
from trained_probe import TrainedDecoder,LMHead

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--dtype',default='fp16')
p.add_argument('--case',help='result directory suffix used by run_full_model.py')
args=p.parse_args();torch.set_num_threads(2)
manifest=json.loads((args.source/'trained-manifest.json').read_text())
board=args.source/args.dtype/('board'+('-'+args.case if args.case else ''));result=json.loads((board/'tokens.json').read_text())
prompt_path=board/'prompt.json'
prompt=json.loads((prompt_path if prompt_path.exists() else args.source/'prompt.json').read_text())
tokens=prompt['token_ids']+result['tokens'][:-1]
reader=gguf.GGUFReader(args.model)
t=next(t for t in reader.tensors if t.name=='token_embd.weight')
embedding=gguf.dequantize(t.data,t.tensor_type)
hidden=torch.from_numpy(embedding[tokens][None].copy())
with torch.inference_mode():
    for index in range(manifest['language_layers_total']):
        layer=TrainedDecoder(reader,index).eval();hidden=layer.full(hidden)
        print('reference layer',index,flush=True)
    expected=LMHead(reader,layer.eps).eval()(hidden[:,len(prompt['token_ids'])-1:]).numpy()[0]
actual=np.fromfile(board/'logits.f32',np.float32).reshape(-1,manifest['vocab_size'])
assert actual.shape==expected.shape and len(actual)==result['generated_tokens']
assert np.isfinite(actual).all() and np.isfinite(expected).all()
relative=np.linalg.norm(actual-expected,axis=1)/np.linalg.norm(expected,axis=1)
cosine=(actual*expected).sum(-1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))
top=expected.argmax(-1);predicted=actual.argmax(-1)
np.testing.assert_array_equal(predicted,result['tokens'])
tokenizer=Tokenizer.from_file(str(args.source/'tokenizer.json'))
summary={'scope':'all decoder layers and final LM head; logits checked on actual NPU-generated history',
         'model':args.model.name,'decoder_layers':manifest['language_layers_total'],
         'npu_dtype':args.dtype,'prompt':prompt['rendered_prompt'],
         'npu_answer':tokenizer.decode(result['tokens'],skip_special_tokens=False),
         'generated_token_count':len(actual),'max_abs_logit_error':float(np.max(np.abs(actual-expected))),
         'relative_l2_all_logits':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
         'max_step_relative_l2':float(relative.max()),'min_step_cosine':float(cosine.min()),
         'argmax_matches_fp32':int((top==predicted).sum()),'argmax_match_fraction':float((top==predicted).mean()),
         'npu_tokens':predicted.tolist(),'fp32_greedy_tokens_on_npu_history':top.tolist(),
         'step_relative_l2':relative.tolist(),'runtime':result}
reference=board/'cpu-full-reference.json'
if not reference.exists():reference=args.source/'cpu-full-reference.json'
if reference.exists():
    cpu=json.loads(reference.read_text());summary['llama_cpu_reference']=cpu
    assert cpu['response']['prompt']==prompt['rendered_prompt'], 'CPU and NPU prompts differ'
    if 'prompt_token_ids' in cpu:
        assert cpu['prompt_token_ids']==prompt['token_ids']
    summary['llama_cpu_answer']=cpu['response'].get('content')
    summary['tokenizer_independent_match']=cpu['tokenizer_exact_match']
(board/'full-validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
np.save(board/'fp32-reference-logits.npy',expected)
print(json.dumps({k:v for k,v in summary.items() if k not in ('runtime','llama_cpu_reference')},indent=2,ensure_ascii=False))
