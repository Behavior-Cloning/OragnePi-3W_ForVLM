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
p.add_argument('--reference-weights',choices=['original','exported-int8','exported'],default='original')
args=p.parse_args();torch.set_num_threads(2)
manifest=json.loads((args.source/'trained-manifest.json').read_text())
board=args.source/args.dtype/('board'+('-'+args.case if args.case else ''));result=json.loads((board/'tokens.json').read_text())
fusion=result.get('fusion_width',1)
runner_config=json.loads((board/'runner-config.json').read_text()) if (board/'runner-config.json').exists() else {}
if result.get('prefill_batch_size',0):
    assert runner_config.get('prefill_variant'), 'missing batched weight provenance'
prompt_path=board/'prompt.json'
prompt=json.loads((prompt_path if prompt_path.exists() else args.source/'prompt.json').read_text())
tokens=prompt['token_ids']+(result['feedback_tokens'] if result.get('fixed_feedback') else result['tokens'])[:-1]
reader=gguf.GGUFReader(args.model)
t=next(t for t in reader.tensors if t.name=='token_embd.weight')
embedding=gguf.dequantize(t.data,t.tensor_type)
hidden=torch.from_numpy(embedding[tokens][None].copy())
trace_path=board/'hidden-trace.f32'
trace=np.fromfile(trace_path,np.float32).reshape(len(tokens),manifest['language_layers_total'],manifest['dim']) if trace_path.exists() else None
trace_errors=[];constant_records=[]
with torch.inference_mode():
    for index in range(manifest['language_layers_total']):
        layer=TrainedDecoder(reader,index).eval()
        if args.reference_weights!='original':
            from exported_int8 import apply_exported_int8
            constant_records.append({'layer':index,'constants':apply_exported_int8(layer,
                args.source/args.dtype/f'l{index//fusion*fusion}',prefix=f'b{index%fusion}_' if fusion>1 else '')})
            if result.get('prefill_batch_size',0):
                batch_layer=TrainedDecoder(reader,index).eval()
                apply_exported_int8(batch_layer,args.source/runner_config['prefill_variant']/f'l{index//fusion*fusion}',
                    prefix=f'b{index%fusion}_' if fusion>1 else '')
                for key,value in layer.state_dict().items():
                    assert torch.equal(value,batch_layer.state_dict()[key]), 'batch/decode constants differ: '+key
        hidden=layer.full(hidden)
        if trace is not None:
            ref=hidden.numpy()[0];actual_hidden=trace[:,index,:]
            relative_hidden=np.linalg.norm(actual_hidden-ref,axis=-1)/np.maximum(np.linalg.norm(ref,axis=-1),1e-20)
            trace_errors.append({'layer':index,'relative_l2':float(np.linalg.norm(actual_hidden-ref)/np.linalg.norm(ref)),
                                 'max_step_relative_l2':float(relative_hidden.max()),'step_relative_l2':relative_hidden.tolist()})
        print('reference layer',index,flush=True)
    head=LMHead(reader,layer.eps).eval()
    if args.reference_weights!='original':
        from exported_int8 import apply_exported_head
        constant_records.append({'head':apply_exported_head(head,args.source,args.dtype,manifest)})
    expected=head(hidden[:,len(prompt['token_ids'])-1:]).numpy()[0]
actual=np.fromfile(board/'logits.f32',np.float32).reshape(-1,manifest['vocab_size'])
assert actual.shape==expected.shape and len(actual)==result['generated_tokens']
assert np.isfinite(actual).all() and np.isfinite(expected).all()
relative=np.linalg.norm(actual-expected,axis=1)/np.linalg.norm(expected,axis=1)
cosine=(actual*expected).sum(-1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))
top=expected.argmax(-1);predicted=actual.argmax(-1)
reference_logp=torch.log_softmax(torch.from_numpy(expected),dim=-1)
actual_logp=torch.log_softmax(torch.from_numpy(actual),dim=-1)
kl=(reference_logp.exp()*(reference_logp-actual_logp)).sum(-1).numpy()
np.testing.assert_array_equal(predicted,result['tokens'])
tokenizer=Tokenizer.from_file(str(args.source/'tokenizer.json'))
summary={'scope':'all decoder layers and final LM head; logits checked on '+('fixed teacher-forced history' if result.get('fixed_feedback') else 'actual NPU-generated history'),
         'fixed_feedback':result.get('fixed_feedback',False),
         'reference_weights':args.reference_weights,
         'exported_constant_scope':'decoder matrices, QKV biases, norm scales and all output-head shards; FP32 arithmetic' if args.reference_weights!='original' else None,
         'constant_reconstruction_version':3 if args.reference_weights!='original' else None,
         'fusion_width':fusion,
         'prefill_decode_constants_identical':True if result.get('prefill_batch_size',0) and args.reference_weights!='original' else None,
         'constant_records':constant_records,
         'model':args.model.name,'decoder_layers':manifest['language_layers_total'],
         'npu_dtype':args.dtype,'prompt':prompt['rendered_prompt'],
         'npu_answer':None if result.get('fixed_feedback') else tokenizer.decode(result['tokens'],skip_special_tokens=False),
         'predictions_text_on_fixed_history':tokenizer.decode(result['tokens'],skip_special_tokens=False) if result.get('fixed_feedback') else None,
         'generated_token_count':len(actual),'max_abs_logit_error':float(np.max(np.abs(actual-expected))),
         'relative_l2_all_logits':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
         'max_step_relative_l2':float(relative.max()),'min_step_cosine':float(cosine.min()),
         'argmax_matches_fp32':int((top==predicted).sum()),'argmax_match_fraction':float((top==predicted).mean()),
         'mean_kl_nats':float(kl.mean()),'max_step_kl_nats':float(kl.max()),
         'npu_tokens':predicted.tolist(),'fp32_greedy_tokens_on_npu_history':top.tolist(),
         'step_relative_l2':relative.tolist(),'runtime':result}
summary['execution_gate_v1']={
    'applicable':args.reference_weights!='original',
    'relative_l2_limit':0.05,'min_cosine_limit':0.99,'argmax_match_min':0.95,
    'passed':bool(summary['relative_l2_all_logits']<=0.05 and cosine.min()>=0.99 and (top==predicted).mean()>=0.95)
              if args.reference_weights!='original' else None,
    'scope':'execution on this trajectory only; not holdout quality acceptance'}
if trace is not None:summary['hidden_trace_errors']=trace_errors
reference=board/'cpu-full-reference.json'
if not reference.exists():
    candidate=args.source/'cpu-full-reference.json'
    if candidate.exists() and json.loads(candidate.read_text())['response']['prompt']==prompt['rendered_prompt']:
        reference=candidate
if reference.exists():
    cpu=json.loads(reference.read_text());summary['llama_cpu_reference']=cpu
    assert cpu['response']['prompt']==prompt['rendered_prompt'], 'CPU and NPU prompts differ'
    if 'prompt_token_ids' in cpu:
        assert cpu['prompt_token_ids']==prompt['token_ids']
    summary['llama_cpu_answer']=cpu['response'].get('content')
    summary['tokenizer_independent_match']=cpu['tokenizer_exact_match']
suffix='-w8' if args.reference_weights!='original' else ''
(board/('full-validation'+suffix+'.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
np.save(board/('fp32-reference-logits'+suffix+'.npy'),expected)
print(json.dumps({k:v for k,v in summary.items() if k not in ('runtime','llama_cpu_reference')},indent=2,ensure_ascii=False))
