"""Expand complete-vocabulary grouped-head A/B after aligned matrix acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true',help='wait for each baseline case; no board lock while waiting')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent;project=scripts.parents[1]
    source=project/'npu/results/full-intern1b-sharded-v2-c128'
    state_path=a.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'cases':{},'status':'running'}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    for history in (512,1024):
        for new in (32,64,128):
            name=f'intern1b-h{history}-n{new}'
            if state['cases'].get(name,{}).get('status')=='complete':continue
            deadline=time.monotonic()+14400
            while True:
                matrix=json.loads((a.matrix/'state.json').read_text());entry=matrix['cases'].get(name,{})
                if matrix.get('status')=='blocked-storage' or entry.get('status')=='failed':
                    raise RuntimeError('baseline blocked/failed; inspect before head experiments')
                if entry.get('status')=='quality-failed':
                    state['cases'][name]={'status':'blocked-quality','reason':'baseline execution gate failed; no speed experiment',
                                          'reference':entry.get('reference')}
                    save();break
                if entry.get('bands',{}).get('medium',{}).get('status')=='complete':break
                if not a.wait or time.monotonic()>deadline:raise RuntimeError('baseline not ready: '+name)
                if matrix.get('status')!='running':raise RuntimeError('matrix stopped before '+name)
                state.update(status='waiting-for-baseline',waiting_case=name);save();time.sleep(15)
            if entry.get('status')=='quality-failed':continue
            reference=Path(entry['reference']);gate=json.loads((reference/'full-validation-w8.json').read_text())
            assert gate['execution_gate_v1']['passed']
            r=gate['runtime'];assert r['decoder_layers']==24 and r['head_on_npu'] and r['generated_tokens']==128
            assert r['cached_prefix_tokens']==history and r['new_prompt_tokens']==new
            case=reference.name.removeprefix('board-');tag=entry['tag']
            output=a.output/(name+'-head-ab.json')
            if output.exists():
                report=json.loads(output.read_text())
                assert report['status']=='complete','preserve failed or incomplete attempt'
            else:
                state.update(status='running',waiting_case=None);state['cases'][name]={'status':'running'};save()
                command=[sys.executable,str(scripts/'benchmark_variants.py'),'--source',str(source),
                    '--reference',str(reference),'--variant','serial',tag,case,'12','2',
                    '--variant','grouped',tag,case,'12','2',
                    '--variant-reference','serial',str(reference),'--variant-reference','grouped',str(reference),
                    '--prefill','serial','32','--prefill','grouped','32',
                    '--fast-greedy','serial','--fast-greedy','grouped','--group-head','grouped',
                    '--fixed-feedback','--cpu-affinity','6,7','--max-start-temp-mc','68000',
                    '--temperature-band-mc','55000','64999','--repeats','5',
                    '--experiment','headmatrix-'+name,'--output',str(output)]
                print(name,'complete-depth head A/B',flush=True)
                with (a.output/(name+'.log')).open('x') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                report=json.loads(output.read_text())
            diagnostics=[r for r in report['runs'] if r['phase']=='diagnostic']
            assert len(diagnostics)==2 and len({r['logits_sha256'] for r in diagnostics})==1
            med=report['medians'];pairs=[]
            for repeat in range(5):
                pair={r['label']:r['result'] for r in report['runs'] if r['phase']=='measure' and r['repeat']==repeat}
                assert set(pair)=={'serial','grouped'}
                pairs.append(100*(pair['grouped']['decode_tokens_per_second']/pair['serial']['decode_tokens_per_second']-1))
            state['cases'][name]={'status':'complete','reference':str(reference),'result':str(output),
                'serial':med['serial'],'grouped':med['grouped'],'paired_gains_percent':pairs,
                'logits_identical':True,'gate':gate['execution_gate_v1'],
                'decode_gain_percent':100*(med['grouped']['decode_tokens_per_second']/med['serial']['decode_tokens_per_second']-1)}
            save();print(name,state['cases'][name]['decode_gain_percent'],flush=True)
    state['status']='partial-quality-blocked' if any(c['status']=='blocked-quality' for c in state['cases'].values()) else 'complete'
    state['waiting_case']=None;save()


if __name__=='__main__':main()
