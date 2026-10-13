"""Full-depth padded causal tail vs scalar tail, with its own CPU trajectory.

Extra short-turn experiment; never substitute its 20/12 new tokens for the
required 32/64/128 matrix. Keep the existing deployed graph weights unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('smol500','intern1b'),required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-scalar-trial',type=Path,
                   help='reuse a completed scalar diagnostic and CPU trajectory in a new attempt')
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    scripts=Path(__file__).resolve().parent;root=scripts.parents[1]
    smol=a.model=='smol500';model_id='smolvlm-500m' if smol else 'internvl3-1b'
    source=root/'npu/results'/('full-smol500-stable-v3-c128' if smol else 'full-intern1b-sharded-v2-c128')
    model=root/'models'/model_id/('SmolVLM-500M-Instruct-Q8_0.gguf' if smol else 'InternVL3-1B-Instruct-Q8_0.gguf')
    dtype='w8-fused2-c768-dfpgateup' if smol else 'w8-fused2-c768-dfpgateup-mse-v2'
    prefill='w8-fused2-c768-t32-dfpgateup' if smol else 'w8-fused2-c768-t16-dfpgateup-mse'
    tag='full-smol500-c768-dfpgateup' if smol else 'full-intern1b-c768-dfpmse-prefill16'
    group=16 if smol else 12;batch=32 if smol else 16;minimum=16 if smol else 8
    fixture=next(c for c in json.loads((a.fixtures/'matrix.json').read_text()) if c['case'].startswith(a.model+'-'))
    case='tail-'+fixture['case'];state={'status':'cpu-reference','scope':__doc__,'fixture':fixture,
        'candidate_tail_min':minimum,'independent_quality_accepted':False,'variants':{}}
    def save():
        tmp=a.output/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'state.json')
    def run(step,args):
        with (a.output/(step+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(scripts/args[0]),*map(str,args[1:])],
                           cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        previous=None
        if a.reuse_scalar_trial:
            prior_state=a.reuse_scalar_trial/'state.json'
            previous=json.loads(prior_state.read_text())
            assert previous['status']=='development-quality-failed'
            assert previous['fixture']==fixture and previous['candidate_tail_min']==minimum
            assert set(previous['variants'])=={'scalar'}, 'reuse only the completed scalar-only diagnostic'
            cpu=a.reuse_scalar_trial/'cpu'
            assert (cpu/'tokens.json').is_file()
            state['reused_scalar_trial']={'path':str(a.reuse_scalar_trial.resolve()),
                'state_sha256':hashlib.sha256(prior_state.read_bytes()).hexdigest(),
                'cpu_tokens_sha256':hashlib.sha256((cpu/'tokens.json').read_bytes()).hexdigest()}
        else:
            cpu=a.output/'cpu'
            run('cpu',['run_cpu_reference.py','--prompt-folder',fixture['folder'],'--model-id',model_id,
                '--context',768,'--tag',case+'-cpu','--output',cpu,'--repeats',5,
                '--threads',2,'--cpu-affinity','6,7','--flash-attn','off' if smol else 'auto',
                '--max-start-temp-mc',50000,'--temperature-band-mc',55000,64999])
        refs={}
        for label in ('scalar','padded'):
            if previous and label=='scalar':
                old=previous['variants']['scalar'];ref=Path(old['reference'])
                assert ref.parent.resolve()==(source/dtype).resolve()
                runtime=json.loads((ref/'tokens.json').read_text())
                assert runtime['decoder_layers']==(32 if smol else 24) and runtime['generated_tokens']==128
                assert runtime['cached_prefix_tokens']==512 and runtime['new_prompt_tokens']==fixture['new_tokens']
                assert runtime['prefill_tail_min']==0 and runtime['fixed_feedback']
                assert hashlib.sha256((Path(fixture['folder'])/'prompt.i32').read_bytes()).hexdigest()==fixture['prompt_sha256']
                refs[label]=ref;state['variants'][label]=old;save();continue
            name=case+'-'+label;ref=source/dtype/('board-'+name);refs[label]=ref
            assert not ref.exists(), 'preserve previous full-run evidence'
            state['status']='full-depth-'+label;save()
            cmd=['run_full_model.py','--source',source,'--dtype',dtype,'--tag',tag,'--case',name,
                 '--prompt-folder',fixture['folder'],'--max-new',128,'--cached-prefix',512,
                 '--decoder-group',group,'--fusion-width',2,'--prefill-variant',prefill,
                 '--fast-greedy','--cpu-affinity','6,7','--skip-deploy','--feedback-from',cpu/'tokens.json']
            if not smol:cmd+=['--group-head']
            if label=='padded':cmd+=['--prefill-tail-min',minimum]
            run(label+'-full',cmd)
            for weights in ('exported','original'):
                run(label+'-'+weights,['validate_full.py','--source',source,'--model',model,'--dtype',dtype,
                    '--case',name,'--reference-weights',weights])
            exported=json.loads((ref/'full-validation-w8.json').read_text())
            original=json.loads((ref/'full-validation.json').read_text())
            quality={'reference':str(ref),'execution_gate':exported['execution_gate_v1'],
                     'mean_kl_nats':original['mean_kl_nats'],
                     'perplexity_ratio':original['fixed_trajectory_nll']['perplexity_ratio']}
            state['variants'][label]=quality;save()
        # A rejected scalar baseline must not prevent us from learning whether
        # the padded path is numerically valid. Neither can enter formal A/B
        # timing until BOTH full-model diagnostics meet the frozen gates.
        if any(not v['execution_gate']['passed'] or v['mean_kl_nats']>.05 or
               v['perplexity_ratio']>1.05 for v in state['variants'].values()):
            state['status']='development-quality-failed';save();return
        state['status']='five-repeat-performance';save()
        cmd=['benchmark_variants.py','--source',source,'--reference',refs['scalar'],
             '--fixed-feedback','--cpu-affinity','6,7','--temperature-band-mc',55000,64999,
             '--max-start-temp-mc',68000,'--repeats',5,'--experiment',a.output.name,
             '--output',a.output/'performance.json','--prefill-tail-min','padded',minimum]
        for label in ('scalar','padded'):
            cmd+=['--variant',label,tag,case+'-'+label,group,2,'--variant-reference',label,refs[label],
                  '--prefill',label,batch,'--fast-greedy',label]
            if not smol:cmd+=['--group-head',label]
        run('performance',cmd)
        result=json.loads((a.output/'performance.json').read_text());assert result['status']=='complete'
        state.update(status='complete',medians=result['medians']);save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
