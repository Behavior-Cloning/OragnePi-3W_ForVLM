"""Resume the exact-length full-depth matrix; retain failed cases and logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['cpu','npu'],required=True)
    p.add_argument('--case',action='append',default=[])
    p.add_argument('--run-id',default='aligned2')
    a=p.parse_args()
    assert a.run_id.replace('-','').isalnum()
    matrix=json.loads((a.root/'matrix.json').read_text())
    assert matrix['generated_steps']==128 and matrix['repeats']==5 and len(matrix['cases'])==12
    scripts=Path(__file__).resolve().parent
    project=scripts.parents[1]
    os.chdir(project)
    states=a.root/('state-'+a.stage+'.json')
    state=json.loads(states.read_text()) if states.exists() else {'stage':a.stage,'cases':{}}
    def save():
        temporary=states.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(states)
    def run(case,step,command):
        log=a.root/(case+'-'+step+'.log')
        assert not log.exists(), 'preserve earlier attempt: '+str(log)
        print(case,step,flush=True)
        with log.open('w') as output:
            result=subprocess.run([sys.executable,str(scripts/command[0]),*map(str,command[1:])],
                                  stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{step} exited {result.returncode}; see {log}')
    for c in matrix['cases']:
        case=c['case']
        if a.case and case not in a.case:continue
        if state['cases'].get(case,{}).get('status')=='complete':continue
        if case in state['cases']:raise RuntimeError('review incomplete attempt before resuming: '+case)
        row={'status':'running','started':time.time()};state['cases'][case]=row;save()
        try:
            prompt=Path(c['folder']);cpu=a.root/(case+'-cpu')
            if a.stage=='cpu':
                run(case,'cpu',['run_cpu_reference.py','--prompt-folder',prompt,'--model-id',c['model_id'],
                    '--context',c['capacity'],'--tag',a.run_id+'-'+case+'-cpu','--output',cpu,'--repeats',5])
                r=json.loads((cpu/'results.json').read_text())
                assert r['status']=='complete' and r['cached_prefix']==c['history']
                assert len(r['runs'])==6 and len(json.loads((cpu/'tokens.json').read_text())['tokens'])==128
                row['medians']=r['medians']
            else:
                deadline=time.monotonic()+7200
                while not (cpu/'tokens.json').exists():
                    cpu_states=a.root/'state-cpu.json'
                    if cpu_states.exists():
                        status=json.loads(cpu_states.read_text())['cases'].get(case,{}).get('status')
                        if status=='failed':raise RuntimeError('CPU case failed; inspect paired evidence')
                    if time.monotonic()>deadline:raise TimeoutError('CPU trajectory pending')
                    time.sleep(5)
                assert json.loads((cpu/'results.json').read_text())['status']=='complete'
                smol=c['model']=='smol500';cap=c['capacity']
                source=project/'npu/results'/('full-smol500-stable-v3-c128' if smol else 'full-intern1b-sharded-v2-c128')
                suffix='splitkv-transk' if smol else 'splitkv-i16mm'
                decode=f'w8-fused2-c{cap}-{suffix}'
                prefill=f'w8-fused2-c{cap}-t32-{suffix}'
                for variant in (decode,prefill):
                    manifest=source/variant/'stack-conversion.json'
                    deadline=time.monotonic()+3600
                    while not manifest.exists():
                        if time.monotonic()>deadline:raise TimeoutError('compile pending: '+str(manifest))
                        time.sleep(5)
                    records=json.loads(manifest.read_text())
                    assert records and all(r['exit_code']==0 for r in records)
                tag=f'full-{c["model"]}-c{cap}-aligned32'
                reference=source/decode/('board-'+a.run_id+'-'+case)
                row.update(reference=str(reference),tag=tag,decode=decode,prefill=prefill);save()
                deployment=a.root/(tag+'-deployed.json')
                command=['run_full_model.py','--source',source,'--dtype',decode,'--tag',tag,
                    '--case',a.run_id+'-'+case,'--prompt-folder',prompt,'--max-new',128,
                    '--cached-prefix',c['history'],'--decoder-group',16 if smol else 12,
                    '--fusion-width',2,'--prefill-variant',prefill,'--fast-greedy',
                    '--cpu-affinity','6,7','--feedback-from',cpu/'tokens.json']
                existing=project/'npu/results/aligned-20261013'/deployment.name
                old_reference=source/decode/('board-aligned-'+case)
                old_deployed=(old_reference/'exit.json').exists() and json.loads((old_reference/'exit.json').read_text())['exit_code']==0
                if deployment.exists() or existing.exists() or old_deployed:command+=['--skip-deploy']
                elif not smol or cap==1280:
                    command+=['--reuse-tag',f'full-{c["model"]}-c{cap}-{suffix}']
                else:
                    # Existing ordinary-K stack supplies identical embedding/head;
                    # all transposed decoder graphs are new and uploaded.
                    command+=['--reuse-tag',f'full-smol500-c{cap}-splitkv-i16mm',
                              '--deploy-layers',*range(0,32,2)]
                run(case,'npu-diagnostic',command)
                deployment.write_text(json.dumps({'tag':tag,'decode':decode,'prefill':prefill})+'\n')
                run(case,'validate',['validate_full.py','--source',source,'--dtype',decode,
                    '--case',a.run_id+'-'+case,'--model',project/json.loads((prompt/'prompt.json').read_text())['model_path'],
                    '--reference-weights','exported'])
                gate=json.loads((reference/'full-validation-w8.json').read_text())
                row['execution_gate']=gate['execution_gate_v1'];save()
                if not gate['execution_gate_v1']['passed']:
                    row['status']='quality-failed';save();continue
                performance=a.root/(case+'-npu-performance.json')
                run(case,'npu-performance',['benchmark_variants.py','--source',source,'--reference',reference,
                    '--variant','batch32',tag,a.run_id+'-'+case,16 if smol else 12,2,
                    '--variant-reference','batch32',reference,'--prefill','batch32',32,
                    '--fast-greedy','batch32','--fixed-feedback','--cpu-affinity','6,7',
                    '--max-start-temp-mc',68000,'--max-response-temp-mc',68000,'--repeats',5,'--experiment',a.run_id+'-'+case+'-v1',
                    '--output',performance])
                r=json.loads(performance.read_text());assert r['status']=='complete'
                row['medians']=r['medians']['batch32']
            row['status']='complete'
        except Exception as exc:
            row.update(status='failed',error=str(exc));print(case,str(exc),flush=True)
        finally:
            row['finished']=time.time();save()
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
