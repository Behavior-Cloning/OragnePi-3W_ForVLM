"""Bounded t8 screening against each current quantized prefill, not full TTFT."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    scripts=Path(__file__).resolve().parent
    root=scripts.parent/'results'
    output=root/'prefill-current-screen-20261014'
    output.mkdir(exist_ok=True)
    state_path=output/'state.json'
    assert not state_path.exists(), 'inspect and preserve any prior attempt'
    state={'status':'running','scope':__doc__,'cases':{},'full_model_speed_claim':False}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    def run(name,command):
        with (output/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(scripts/command[0]),*map(str,command[1:])],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        for model,source_name,model_path,baseline,baseline_size in [
            ('intern1b','full-intern1b-sharded-v2-c128','models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf','w8-fused2-c768-t16-cap1-productwide',16),
            ('smol500','full-smol500-stable-v3-c128','models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf','w8-fused2-c768-t32-dfpgateup',32)]:
            source=root/source_name
            for size in (baseline_size,8):
                name=f'{model}-t{size}';row={'status':'waiting-for-compiled-graph'}
                state['cases'][name]=row;save()
                graph=source/baseline/'l0' if size==baseline_size else output/name
                deadline=time.monotonic()+600
                while not (graph/'conversion.json').exists():
                    assert time.monotonic()<deadline,'compile did not complete';time.sleep(5)
                fixture=output/('fixture-'+name)
                row.update(status='building-independent-fixture',graph=str(graph),fixture=str(fixture));save()
                run('fixture-'+name,['resized_fixture.py','--model',model_path,'--source',source,
                    '--folder',graph,'--prompt-folder',root/f'aligned-20261013r2/{model}-h512-n64',
                    '--history','512','--output',fixture])
                result=output/('probe-'+name)
                row.update(status='waiting-board-probe',result=str(result));save()
                run('probe-'+name,['run_tensor_probe.py','--folder',fixture,
                    '--tag','current-prefill-'+name+'-20261014','--repeats','50','--warmup','3',
                    '--cpu-affinity','6,7','--result-dir',result])
                d=json.loads((result/'board-validation.json').read_text())
                row.update(status='complete',gate=d['gate'],timing=d['timing']);save()
        state['status']='complete';save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
