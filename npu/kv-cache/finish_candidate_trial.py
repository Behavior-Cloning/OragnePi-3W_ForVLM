"""Finish a queued full-depth run, gate it, then collect five clean measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--dtype',required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--baseline-tag')
    p.add_argument('--group-head',action='store_true')
    a=p.parse_args();assert bool(a.baseline)==bool(a.baseline_tag)
    scripts=Path(__file__).resolve().parent
    a.output.mkdir(parents=True,exist_ok=True)
    state_path=a.output/'state.json'
    state={'status':'waiting-for-full-run','independent_quality_accepted':False}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    def run(step,command):
        with (a.output/(step+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(scripts/command[0]),*map(str,command[1:])],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    candidate=a.source/a.dtype/('board-'+a.case)
    try:
        save();deadline=time.monotonic()+10800
        while not (candidate/'tokens.json').exists():
            if (candidate/'exit.json').exists():
                assert json.loads((candidate/'exit.json').read_text())['exit_code']==0,'full runner failed'
            assert time.monotonic()<deadline,'full runner did not finish before deadline'
            time.sleep(15)
        runtime=json.loads((candidate/'tokens.json').read_text())
        manifest=json.loads((a.source/'trained-manifest.json').read_text())
        assert runtime['decoder_layers']==manifest['language_layers_total']
        assert runtime['generated_tokens']==128 and runtime['head_on_npu'] and runtime['fixed_feedback']
        state['status']='validating-full-logits';save()
        for weights,suffix in [('exported','-w8'),('original','')]:
            if not (candidate/('full-validation'+suffix+'.json')).exists():
                run(weights,['validate_full.py','--model',a.model,'--source',a.source,'--dtype',a.dtype,
                             '--case',a.case,'--reference-weights',weights])
        validation=json.loads((candidate/'full-validation-w8.json').read_text())
        original=json.loads((candidate/'full-validation.json').read_text())
        state.update(execution_gate=validation['execution_gate_v1'],
                     original_q8_development={'l2':original['relative_l2_all_logits'],
                         'mincos':original['min_step_cosine'],'mean_kl':original['mean_kl_nats'],
                         'perplexity_ratio':original['fixed_trajectory_nll']['perplexity_ratio']})
        if not validation['execution_gate_v1']['passed']:
            state['status']='quality-failed';save();return
        if original['fixed_trajectory_nll']['perplexity_ratio']>1.05 or original['mean_kl_nats']>.05:
            state['status']='development-quality-failed';save();return
        state['status']='five-repeat-performance';save()
        variants=[('candidate',a.tag,a.case,candidate)]
        if a.baseline:variants.insert(0,('baseline',a.baseline_tag,a.baseline.name.removeprefix('board-'),a.baseline))
        reference=a.baseline or candidate
        output=a.output/'performance.json'
        command=['benchmark_variants.py','--source',a.source,'--reference',reference]
        for label,tag,case,path in variants:
            r=json.loads((path/'tokens.json').read_text())
            command+=['--variant',label,tag,case,str(r['decoder_group_size']),str(r['fusion_width']),
                      '--variant-reference',label,path,'--prefill',label,str(r['prefill_batch_size']),
                      '--fast-greedy',label]
            if a.group_head:command+=['--group-head',label]
        command+=['--fixed-feedback','--cpu-affinity','6,7','--max-start-temp-mc','68000',
                  '--temperature-band-mc','55000','64999','--repeats','5',
                  '--experiment',a.output.name,'--output',output]
        if not output.exists():run('performance',command)
        result=json.loads(output.read_text());assert result['status']=='complete'
        state.update(status='complete-development-trial',medians=result['medians'],performance=str(output))
        save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
