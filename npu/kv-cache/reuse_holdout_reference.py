"""Bind immutable standard-Q8 reference logits to another candidate of that model.

This is repeated frozen-suite regression, not a newly created independent set.
Reference logits/feedback never come from the NPU candidate being evaluated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    original=args.reference.resolve();candidate=args.candidate.resolve()
    raw=(original/'state.json').read_bytes();state=json.loads(raw)
    assert state['status']=='complete-reference-only' and len(state['cases'])==36
    prior=Path(state['candidate']).resolve()
    old=json.loads((prior.parents[1]/'trained-manifest.json').read_text())
    new=json.loads((candidate.parents[1]/'trained-manifest.json').read_text())
    assert old['sha256']==new['sha256']==digest(Path(state['model']))
    assert old['language_layers_total']==new['language_layers_total']==state['layers']
    assert old['vocab_size']==new['vocab_size']==state['vocab']
    assert digest(prior/'full-validation-w8.json')==state['candidate_execution_validation_sha256']
    validation=candidate/'full-validation-w8.json'
    assert json.loads(validation.read_text())['execution_gate_v1']['passed']
    runtime=json.loads((candidate/'tokens.json').read_text())
    suite=Path(state.get('holdout_path',Path(__file__).with_name('quality-holdout-v1.json')))
    assert digest(suite)==state['holdout_sha256']
    assert all(row['prompt_count']+row['generated_count']-1<=runtime['capacity'] for row in state['cases'])
    verified=[]
    for row in state['cases']:
        folder=original/row['case'];logits=folder/'reference-logits.f32'
        assert logits.stat().st_size==row['generated_count']*state['vocab']*4
        assert digest(logits)==row['logits_sha256']
        request=json.loads((folder/'request.json').read_text())
        assert request==row
        verified.append((row['case'],logits,folder/'request.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    for name,logits,request in verified:
        target=args.output/name;target.mkdir()
        for path in (logits,request):os.link(path,target/path.name)
    state.update(candidate=str(candidate),candidate_execution_validation_sha256=digest(validation),
                 model_sha256=new['sha256'],holdout_path=str(suite.resolve()),
                 reused_standard_reference={'path':str(original),'state_sha256':hashlib.sha256(raw).hexdigest(),
                    'original_candidate':str(prior),'reference_logits_sha256_verified':True,
                    'scope':'repeated frozen-suite regression; not a fresh independent holdout'})
    (args.output/'state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n')
    print('verified and reused 36 standard-Q8 reference trajectories; no NPU reference feedback')


if __name__=='__main__':main()
