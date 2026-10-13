"""Freeze independent Q8-dequantized FP32 quality trajectories on workstation.

This is a standard-library numerical anchor, not board CPU performance. The
holdout is consumed for acceptance only; do not tune on failures without
retiring its version. Each model keeps all language layers and its whole head.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import gguf
import numpy as np
import torch
from resident_http import ChatCodec
from validate_standard_oracle import load_standard


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--suite',type=Path,default=Path(__file__).with_name('quality-holdout-v1.json'))
    a=p.parse_args()
    assert not a.output.exists() and 1<=a.steps<=128
    a.output.mkdir(parents=True)
    holdout=a.suite
    suite=json.loads(holdout.read_text())
    assert len(suite['cases'])==12 and len({c['id'] for c in suite['cases']})==12
    assert all(len(c['turns'])==3 for c in suite['cases'])
    manifest=json.loads((a.source/'trained-manifest.json').read_text())
    gate=json.loads((a.candidate/'full-validation-w8.json').read_text())
    assert gate['execution_gate_v1']['passed']
    torch.set_num_threads(4)
    reader=gguf.GGUFReader(a.model)
    model,config=load_standard(reader)
    assert len(model.model.layers)==manifest['language_layers_total']
    system='You are a friendly desktop companion. Reply briefly and naturally.'
    codec=ChatCodec(a.model,system)
    report={'status':'running','scope':'standard Transformers FP32 on dequantized Q8; workstation quality reference, no speed comparison',
            'holdout_sha256':hashlib.sha256(holdout.read_bytes()).hexdigest(),'holdout_version':suite['version'],
            'holdout_path':str(holdout.resolve()),'model':str(a.model),
            'candidate':str(a.candidate),'candidate_execution_validation_sha256':hashlib.sha256((a.candidate/'full-validation-w8.json').read_bytes()).hexdigest(),
            'layers':len(model.model.layers),'vocab':config.vocab_size,'system':system,'steps_limit':a.steps,
            'gate':{'mean_kl_nats_max':.05,'perplexity_ratio_max':1.05,'human_review_required':True},'cases':[]}
    def save():
        (a.output/'state.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        for case in suite['cases']:
            messages=[]
            for turn,text in enumerate(case['turns']):
                messages.append({'role':'user','content':text})
                tokens=codec.encode(messages)
                assert len(tokens)+a.steps-1<=1280
                start=time.monotonic()
                generated=[]
                logits=[]
                with torch.inference_mode():
                    step=model.model(torch.tensor([tokens]),use_cache=True)
                    cache=step.past_key_values
                    for i in range(a.steps):
                        prediction=model.lm_head(step.last_hidden_state[:,-1:])[0,0]
                        logits.append(prediction.numpy().copy())
                        token=int(prediction.argmax())
                        generated.append(token)
                        if token in manifest['eos_token_ids']:
                            break
                        if i+1<a.steps:
                            step=model.model(torch.tensor([[token]]),past_key_values=cache,use_cache=True)
                            cache=step.past_key_values
                name=case['id']+f'-turn{turn+1}'
                folder=a.output/name
                folder.mkdir()
                values=np.stack(logits).astype(np.float32)
                values.tofile(folder/'reference-logits.f32')
                answer=codec.decode(generated)
                row={'case':name,'messages':list(messages),'prompt_tokens':tokens,'feedback_tokens':generated,
                     'reference_answer':answer,'checks':case['checks'],'reference_compute_seconds':time.monotonic()-start,
                     'logits_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'prompt_count':len(tokens),
                     'generated_count':len(generated),'ended_eos':generated[-1] in manifest['eos_token_ids']}
                (folder/'request.json').write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n')
                report['cases'].append(row)
                save()
                messages.append({'role':'assistant','content':answer})
                print(name,len(tokens),len(generated),flush=True)
        report['status']='complete-reference-only'
        save()
    except BaseException as exc:
        report.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
