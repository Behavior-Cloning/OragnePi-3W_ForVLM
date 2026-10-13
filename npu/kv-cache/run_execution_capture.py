"""Two targeted full-depth diagnostic replays of the failed development scene."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    scripts=Path(__file__).resolve().parent;root=scripts.parents[1]
    source=root/'npu/results/full-intern1b-sharded-v2-c128'
    dtype='w8-fused2-c1280-decode-dfpmse-productwide'
    prefix='dfpmse-productwide-intern1b-h1024-n64'
    baseline=source/dtype/('board-'+prefix)
    output=root/'npu/results/intern1b-1024n64-captured-ranges-20261013'
    output.mkdir(exist_ok=False)
    state={'status':'running','scope':__doc__,'captures':[]}
    def save():
        temp=output/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(output/'state.json')
    def run(label,args):
        with (output/(label+'.log')).open('x') as stream:
            subprocess.run([sys.executable,str(scripts/args[0]),*map(str,args[1:])],cwd=root,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        # Prediction k is produced by input position prompt_count+k-1.
        for position,block in ((1112,1),(1196,11)):
            label=f'p{position}b{block}';case=prefix+'-'+label
            run(label,['run_full_model.py','--source',source,'--dtype',dtype,
                '--tag','full-intern1b-c1280-dfpmse-productwide','--case',case,
                '--prompt-folder',root/'npu/results/aligned-20261013r2/intern1b-h1024-n64',
                '--max-new',128,'--cached-prefix',1024,'--fusion-width',2,
                '--prefill-variant','w8-fused2-c1280-t16-dfpmse-productwide',
                '--fast-greedy','--group-head','--cpu-affinity','6,7','--skip-deploy',
                '--feedback-from',root/'npu/results/thermal-aligned-20261013/intern1b-h1024-n64-low-cpu/tokens.json',
                '--capture-decode',position,block])
            folder=source/dtype/('board-'+case)
            assert (folder/'logits.f32').read_bytes()==(baseline/'logits.f32').read_bytes(), 'ungrouped capture changed complete logits'
            fixture=output/(label+'-fixture')
            run(label+'-reference',['captured_fixture.py','--model',root/'models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf',
                '--graph-folder',source/dtype/f'l{block*2}',
                '--capture',folder/'decode-capture-input.f32','--output',fixture])
            run(label+'-ranges',['audit_captured_ranges.py','--fixture',fixture,'--output',output/(label+'-ranges.json')])
            state['captures'].append({'position':position,'block':block,'folder':str(folder),'fixture':str(fixture),
                                       'ranges':str(output/(label+'-ranges.json')),'full_logits_identical':True})
            save()
        state['status']='complete';save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
