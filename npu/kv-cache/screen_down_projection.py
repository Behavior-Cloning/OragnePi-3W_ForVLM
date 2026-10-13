"""Development-only down-projection selection on a complete frozen trajectory.

Reconstruct current exported weights, replay every layer, and isolate each
ordinary down matrix using the same input activation. No board speed or
independent quality claim follows from this FP32 host screening.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from exported_int8 import apply_exported_int8, apply_exported_head
from trained_probe import TrainedDecoder, LMHead


def weight_dfp(matrix):
    peak_fl=math.floor(math.log2(127/float(matrix.abs().max())))
    trials=[]
    for fl in range(peak_fl,peak_fl+7):
        quant=(matrix*2.**fl).round().clamp(-128,127)*2.**(-fl)
        trials.append((float((matrix-quant).double().square().mean()),fl))
    _,fl=min(trials)
    return (matrix*2.**fl).round().clamp(-128,127)*2.**(-fl),fl


def metrics(actual, expected, targets):
    logp=actual.log_softmax(-1);ref=expected.log_softmax(-1)
    ix=torch.arange(len(targets));delta=(-logp[ix,targets]+ref[ix,targets]).mean()
    return {'relative_l2':float((actual-expected).norm()/expected.norm()),
            'min_cosine':float(F.cosine_similarity(actual,expected).min()),
            'argmax_match':float((actual.argmax(-1)==expected.argmax(-1)).float().mean()),
            'mean_kl':float((ref.exp()*(ref-logp)).sum(-1).mean()),
            'perplexity_ratio':float(delta.exp())}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--variant',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--max-local-p95',type=float,default=.02)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    assert 'holdout' not in str(a.reference).lower()
    assert 0<a.max_local_p95<=.05
    torch.set_num_threads(2)
    manifest=json.loads((a.source/'trained-manifest.json').read_text())
    runtime=json.loads((a.reference/'tokens.json').read_text())
    prompt=json.loads((a.reference/'prompt.json').read_text())['token_ids']
    assert runtime['fixed_feedback'] and runtime['decoder_layers']==manifest['language_layers_total']
    assert runtime['generated_tokens']==128 and runtime['fusion_width']==2
    feedback=runtime['feedback_tokens'];ids=prompt+feedback[:-1]
    begin=len(prompt)-1
    reader=gguf.GGUFReader(a.model)
    t=next(t for t in reader.tensors if t.name=='token_embd.weight')
    embedding=torch.from_numpy(gguf.dequantize(t.data,t.tensor_type).copy())
    hidden=embedding[ids][None];candidate_hidden=hidden.clone()
    records=[];selected=[]
    with torch.inference_mode():
        for index in range(manifest['language_layers_total']):
            layer=TrainedDecoder(reader,index).eval()
            original_down=layer.down.clone()
            folder=a.source/a.variant/f'l{index//2*2}'
            provenance=apply_exported_int8(layer,folder,prefix=f'b{index%2}_')
            down_type=next(v['dtype'] for v in provenance if v['matrix']=='down')
            # Keep explicitly sensitive FP16 down projections untouched.
            if down_type=='INT8':
                q,k,v=layer.project(hidden,*layer.rope(torch.arange(len(ids))))
                attn=F.scaled_dot_product_attention(q,k.repeat_interleave(layer.heads//layer.kv,1),
                                                   v.repeat_interleave(layer.heads//layer.kv,1),is_causal=True)
                residual=hidden+F.linear(attn.transpose(1,2).reshape(1,-1,layer.dim),layer.o)
                normalized=layer.norm(residual,layer.norm2)
                product=F.silu(F.linear(normalized,layer.gate))*F.linear(normalized,layer.up)
                expected=F.linear(product,layer.down)
                replacement,fl=weight_dfp(original_down)
                local=F.linear(product,replacement)
                ratio=(local-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-20)
                p95=float(torch.quantile(ratio[:,begin:].flatten(),.95))
                choose=p95<=a.max_local_p95
                record={'layer':index,'down_dtype':down_type,'candidate_fl':fl,
                        'prediction_branch_l2_p95':p95,'prediction_branch_l2_max':float(ratio[:,begin:].max()),
                        'all_tokens_branch_l2_p95':float(torch.quantile(ratio.flatten(),.95)),
                        'selected':choose}
                # Guard the decomposition using the unchanged layer implementation.
                baseline=layer.full(hidden)
                torch.testing.assert_close(residual+expected,baseline,rtol=2e-5,atol=1e-5)
            else:
                assert down_type=='FLOAT16'
                choose=False;record={'layer':index,'down_dtype':down_type,'selected':False}
                baseline=layer.full(hidden)
            hidden=baseline
            if choose:
                layer.down.copy_(replacement);selected.append(index)
            candidate_hidden=layer.full(candidate_hidden)
            records.append(record)
            print(index,record.get('prediction_branch_l2_p95'),choose,flush=True)
        head=LMHead(reader,layer.eps).eval()
        apply_exported_head(head,a.source,a.variant,manifest)
        expected=head(hidden[:,begin:])[0]
        candidate=head(candidate_hidden[:,begin:])[0]
        oracle=np.load(a.reference/'fp32-reference-logits-w8.npy')
        np.testing.assert_allclose(expected.numpy(),oracle,rtol=1e-4,atol=1e-4)
        original=torch.from_numpy(np.load(a.reference/'fp32-reference-logits.npy'))
        targets=torch.tensor(feedback)
        report={'scope':__doc__,'source':str(a.source),'variant':a.variant,'model':str(a.model),
                'model_sha256':sha(a.model),'reference':str(a.reference),
                'reference_validation_sha256':sha(a.reference/'full-validation-w8.json'),
                'frozen_prompt_sha256':sha(a.reference/'prompt.json'),
                'decoder_layers':manifest['language_layers_total'],'vocab':manifest['vocab_size'],
                'history_tokens':runtime['cached_prefix_tokens'],'new_tokens':runtime['new_prompt_tokens'],
                'prediction_steps':128,'selection_threshold_local_p95':a.max_local_p95,
                'selected_layers':selected,'local_records':records,
                'baseline_exported_vs_original':metrics(expected,original,targets),
                'candidate_vs_original':metrics(candidate,original,targets),
                'candidate_vs_baseline_exported':metrics(candidate,expected,targets),
                'baseline_reproduces_existing_oracle':True,'npu_tested':False,'speed_measured':False,
                'independent_quality_accepted':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('selected',selected,'full candidate',report['candidate_vs_original'])


if __name__=='__main__':main()
