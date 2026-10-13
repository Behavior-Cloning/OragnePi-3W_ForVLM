"""Full-depth exact-token matrix at three measured response-start temperatures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BANDS={'low':(40000,50000),'medium':(55000,64999),'high':(65000,80000)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',action='append',default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent;project=scripts.parents[1]
    matrix=json.loads((a.fixtures/'matrix.json').read_text())
    state_path=a.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'bands_mC':BANDS,'cases':{},'status':'running'}
    if state.get('status')=='blocked-storage':
        raise RuntimeError('board root TF storage stalled; verify storage recovery before explicitly clearing this status')
    if state['bands_mC']!={k:list(v) for k,v in BANDS.items()}:
        assert not any(e['bands'].get('high',{}).get('status')=='complete' for e in state['cases'].values())
        state.setdefault('protocol_revisions',[]).append({'old_bands_mC':state['bands_mC'],
            'reason':'70 C start could not be reproduced after heater shutdown; no successful high-band results were relabeled. Medium upper bound is now exclusive 65 C.'})
        state['bands_mC']={k:list(v) for k,v in BANDS.items()}
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    def run(case,step,command):
        log=a.output/(case+'-'+step+'.log')
        attempt=1
        while log.exists():
            attempt+=1;log=a.output/(case+'-'+step+f'-attempt{attempt}.log')
        print(case,step,flush=True)
        with log.open('w') as out:
            r=subprocess.run([sys.executable,str(scripts/command[0]),*map(str,command[1:])],stdout=out,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{step}: exit {r.returncode}, see {log}')
    for c in matrix['cases']:
        case=c['case']
        if a.case and case not in a.case:continue
        if state['cases'].get(case,{}).get('status')=='complete':continue
        smol=c['model']=='smol500';cap=c['capacity']
        source=project/'npu/results'/('full-smol500-stable-v3-c128' if smol else 'full-intern1b-sharded-v2-c128')
        suffix='splitkv-transk' if smol else 'splitkv-i16mm'
        decode=f'w8-fused2-c{cap}-{suffix}';prefill=f'w8-fused2-c{cap}-t32-{suffix}'
        tag=f'full-{c["model"]}-c{cap}-aligned32';npu_case='thermal3-'+case
        reference=source/decode/('board-'+npu_case)
        row=state['cases'].get(case,{'status':'running','started':time.time(),'config':c,'reference':str(reference),'tag':tag,'bands':{}})
        row['status']='running'
        reference=Path(row['reference']);npu_case=reference.name.removeprefix('board-')
        state['cases'][case]=row;save()
        try:
            trajectory=None
            prior_gate=reference/'full-validation-w8.json'
            gate=json.loads(prior_gate.read_text())['execution_gate_v1'] if prior_gate.exists() else None
            for band,(low,high) in BANDS.items():
                bandrow=row['bands'].get(band,{'status':'running'});row['bands'][band]=bandrow;save()
                cpu=Path(bandrow.get('cpu_path',str(a.output/(case+'-'+band+'-cpu'))))
                cpu_attempt=1
                while (cpu/'results.json').exists() and json.loads((cpu/'results.json').read_text()).get('status')!='complete':
                    cpu_attempt+=1;cpu=a.output/(case+'-'+band+f'-cpu-attempt{cpu_attempt}')
                bandrow['cpu_path']=str(cpu);save()
                if not (cpu/'results.json').exists():
                    run(case,band+'-cpu',['run_cpu_reference.py','--prompt-folder',c['folder'],
                        '--model-id',c['model_id'],'--context',cap,'--tag','thermal3-'+case+'-'+band+'-cpu'+(f'-attempt{cpu_attempt}' if cpu_attempt>1 else ''),
                        '--output',cpu,'--repeats',5,'--temperature-band-mc',low,high])
                result=json.loads((cpu/'results.json').read_text());assert result['status']=='complete'
                actual=json.loads((cpu/'tokens.json').read_text())['tokens']
                for r in result['runs']:
                    assert low<=r['response_start_temperature_mC']<=high, 'CPU response outside requested band'
                if trajectory is None:trajectory=actual
                else:assert trajectory==actual,'CPU predicted trajectory changed with temperature'
                if bandrow.get('status')=='complete':continue
                bandrow['cpu']=result['medians'];save()
                if gate is None:
                    prior=reference/'full-validation-w8.json'
                    if prior.exists():gate=json.loads(prior.read_text())['execution_gate_v1']
                if gate is None:
                    deploy=a.output/(tag+'-deployed.json')
                    existing=source/decode/('board-aligned2-'+case)/'exit.json'
                    old_existing=source/decode/('board-aligned-'+case)/'exit.json'
                    already=deploy.exists() or any(f.exists() and json.loads(f.read_text())['exit_code']==0 for f in (existing,old_existing))
                    command=['run_full_model.py','--source',source,'--dtype',decode,'--tag',tag,
                        '--case',npu_case,'--prompt-folder',c['folder'],'--max-new',128,'--cached-prefix',c['history'],
                        '--decoder-group',16 if smol else 12,'--fusion-width',2,'--prefill-variant',prefill,
                        '--fast-greedy','--cpu-affinity','6,7','--feedback-from',cpu/'tokens.json']
                    if already:command+=['--skip-deploy']
                    elif not smol or cap==1280:command+=['--reuse-tag',f'full-{c["model"]}-c{cap}-{suffix}']
                    else:command+=['--reuse-tag',f'full-smol500-c{cap}-splitkv-i16mm','--deploy-layers',*range(0,32,2)]
                    run(case,'npu-diagnostic',command)
                    deploy.write_text(json.dumps({'tag':tag,'decode':decode,'prefill':prefill})+'\n')
                    run(case,'validate',['validate_full.py','--source',source,'--dtype',decode,'--case',npu_case,
                        '--model',project/json.loads((Path(c['folder'])/'prompt.json').read_text())['model_path'],
                        '--reference-weights','exported'])
                    validation=json.loads((reference/'full-validation-w8.json').read_text())
                    gate=validation['execution_gate_v1'];row['execution_gate']=gate
                    runtime=validation['runtime']
                    assert runtime['decoder_layers']==(32 if smol else 24) and runtime['head_on_npu']
                    assert runtime['generated_tokens']==128 and runtime['fixed_feedback']
                    save()
                if not gate['passed']:
                    bandrow['status']='npu-quality-failed';save();continue
                performance=Path(bandrow.get('performance',str(a.output/(case+'-'+band+'-npu.json'))))
                if performance.exists() and json.loads(performance.read_text()).get('status')=='complete':
                    bandrow.update(status='complete',npu=json.loads(performance.read_text())['medians']['batch32']);save();continue
                attempt=1
                while performance.exists():
                    attempt+=1;performance=a.output/(case+'-'+band+f'-npu-attempt{attempt}.json')
                if attempt>1:
                    # Refresh host runner/helper only; keep immutable graphs and
                    # verify new diagnostic outputs against the prior reference.
                    cpu_source=Path(row['bands']['low'].get('cpu_path',str(a.output/(case+'-low-cpu'))))
                    rerun=npu_case+f'-runner{attempt}'
                    run(case,band+'-refresh',['run_full_model.py','--source',source,'--dtype',decode,'--tag',tag,
                        '--case',rerun,'--prompt-folder',c['folder'],'--max-new',128,'--cached-prefix',c['history'],
                        '--decoder-group',16 if smol else 12,'--fusion-width',2,'--prefill-variant',prefill,
                        '--fast-greedy','--cpu-affinity','6,7','--feedback-from',cpu_source/'tokens.json','--skip-deploy'])
                    refreshed=source/decode/('board-'+rerun)
                    assert (refreshed/'logits.f32').read_bytes()==(reference/'logits.f32').read_bytes()
                    # Fresh validation includes the updated runtime metadata.
                    run(case,band+'-refresh-validate',['validate_full.py','--source',source,'--dtype',decode,'--case',rerun,
                        '--model',project/json.loads((Path(c['folder'])/'prompt.json').read_text())['model_path'],'--reference-weights','exported'])
                    npu_case=rerun;reference=refreshed;row['reference']=str(reference);save()
                run(case,band+'-npu',['benchmark_variants.py','--source',source,'--reference',reference,
                    '--variant','batch32',tag,npu_case,16 if smol else 12,2,
                    '--variant-reference','batch32',reference,'--prefill','batch32',32,
                    '--fast-greedy','batch32','--fixed-feedback','--cpu-affinity','6,7',
                    '--max-start-temp-mc',68000,'--temperature-band-mc',low,high,'--repeats',5,
                    '--experiment','thermal3-'+case+'-'+band+f'-attempt{attempt}','--output',performance])
                result=json.loads(performance.read_text());assert result['status']=='complete'
                bandrow.update(status='complete',npu=result['medians']['batch32'],performance=str(performance));save()
                subprocess.run([sys.executable,str(scripts/'report_thermal_matrix.py'),'--root',str(a.output)],check=True)
            row['status']='complete' if gate['passed'] else 'quality-failed'
            row.pop('error',None)
        except Exception as exc:
            row.update(status='failed',error=str(exc));print(case,str(exc),flush=True)
        finally:
            row['finished']=time.time();save()
            subprocess.run([sys.executable,str(scripts/'report_thermal_matrix.py'),'--root',str(a.output)],check=True)
    state['status']='complete' if len(state['cases'])==12 and all(r['status']=='complete' for r in state['cases'].values()) else 'partial'
    save();print(json.dumps({k:v['status'] for k,v in state['cases'].items()}),flush=True)


if __name__=='__main__':main()
