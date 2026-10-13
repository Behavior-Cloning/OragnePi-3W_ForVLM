"""Bounded Q8 CPU search, preserving trajectories and every attempted result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from attempt_paths import select_attempt
from matrix_paths import cpu_result_path


THREAD_CONFIGS = [
    ('a76-t2', 2, '6,7'), ('a76-t1', 1, '6'),
    ('mixed-t4', 4, '4-7'), ('mixed-t6', 6, '2-7'), ('all-t8', 8, '0-7'),
    ('mixed-t6-cooled', 6, '2-7'),
    ('all-t2', 2, '0-7'), ('all-t4', 4, '0-7'),
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',default=[])
    p.add_argument('--wait',action='store_true',help='wait for missing completed medium CPU baselines')
    p.add_argument('--retry-failed',action='store_true',help='new attempt paths; never overwrite partial evidence')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parent/'run_cpu_reference.py'
    fixtures=json.loads((a.fixtures/'matrix.json').read_text())['cases']
    selected=[c for c in fixtures if c['new_tokens']==64 and (not a.case or c['case'] in a.case)]
    assert selected and (not a.case or set(a.case)=={c['case'] for c in selected})
    state_path=a.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'scope':'Q8 only; medium start; 2-repeat screening then 5-repeat finalist; greedy trajectory equality is alignment, not holdout quality',
        'cases':{},'status':'running'}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    for case in selected:
        name=case['case'];entry=state['cases'].setdefault(name,{'runs':{},'status':'running'})
        if entry['status']=='complete' and entry.get('search_revision')==3:continue
        basepath=cpu_result_path(a.matrix,name,'medium')
        deadline=time.monotonic()+14400
        while not basepath.exists() or json.loads(basepath.read_text()).get('status')!='complete':
            matrix_state=json.loads((a.matrix/'state.json').read_text())
            if not a.wait or time.monotonic()>deadline or matrix_state.get('status')!='running':
                raise RuntimeError('complete baseline matrix for this case before CPU search: '+name)
            state.update(status='waiting-for-baseline',waiting_case=name);save();time.sleep(15)
            basepath=cpu_result_path(a.matrix,name,'medium')
        state.update(status='running',waiting_case=None);entry.update(status='running');entry.pop('error',None)
        base=json.loads(basepath.read_text());assert base['status']=='complete'
        expected=next(r['tokens'] for r in base['runs'] if r['phase']=='measure')
        def run(label,threads,affinity,repeats=2,threads_batch=None,ubatch=512,flash='auto',fixture=None,affinity_batch=None):
            fixture=fixture or case;target_name=fixture['case']
            key=label if fixture is case else f'n{fixture["new_tokens"]}-'+label
            recorded=entry['runs'].get(key,{})
            if recorded.get('status')=='rejected-thermal-limit':
                prior=json.loads(Path(recorded['path']).read_text())
                assert prior.get('status')=='failed' and any(t.get('temperature_mC',0)>=80000 for t in prior.get('thermal_abort',[]))
                return recorded  # Keep thermal rejection even when it occurred in attempt2+.
            folder=a.output/(target_name+'-'+label)
            result=folder/'results.json'
            logpath=a.output/(target_name+'-'+label+'.log')
            thermal=False
            if result.exists():
                previous=json.loads(result.read_text())
                thermal=previous.get('status')=='failed' and any(t.get('temperature_mC',0)>=80000 for t in previous.get('thermal_abort',[]))
            if not thermal:
                folder,logpath,attempt=select_attempt(folder,'results.json',logpath,a.retry_failed)
                result=folder/'results.json'
            else:attempt=1
            if result.exists():
                data=json.loads(result.read_text())
            else:
                assert not folder.exists(), 'keep partial result directory'
                command=[sys.executable,str(script),'--prompt-folder',fixture['folder'],
                    '--model-id',fixture['model_id'],'--context',str(fixture['capacity']),
                    '--tag','q8search-'+target_name+'-'+label+('' if attempt==1 else f'-attempt{attempt}'),'--output',str(folder),
                    '--repeats',str(repeats),'--threads',str(threads),'--cpu-affinity',affinity,
                    '--ubatch-size',str(ubatch),'--flash-attn',flash,'--temperature-band-mc','55000','64999']
                command+=['--max-start-temp-mc','50000']
                if threads_batch:command+=['--threads-batch',str(threads_batch)]
                if affinity_batch:command+=['--cpu-affinity-batch',affinity_batch]
                entry['active']=label;save();print(name,label,flush=True)
                with logpath.open('x') as log:
                    child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if child.returncode and not result.exists():child.check_returncode()
                data=json.loads(result.read_text())
            if data.get('status')=='failed' and any(t.get('temperature_mC',0)>=80000 for t in data.get('thermal_abort',[])):
                value={'aligned':False,'repeats':repeats,'path':str(result),'status':'rejected-thermal-limit',
                       'quality':'unmeasured after thermal stop','medians':{},'thermal_abort':data['thermal_abort']}
                entry['runs'][key]=value;entry.pop('active',None);save();return value
            assert data['status']=='complete'
            measured=[r for r in data['runs'] if r['phase']=='measure'];assert len(measured)==repeats
            canonical=expected
            if fixture is not case:
                other=json.loads(cpu_result_path(a.matrix,target_name,'medium').read_text())
                assert other['status']=='complete'
                canonical=next(r['tokens'] for r in other['runs'] if r['phase']=='measure')
            aligned=all(r['tokens']==canonical for r in measured)
            value={'config':{'threads':threads,'affinity':affinity,'threads_batch':threads_batch or threads,
                             'ubatch':ubatch,'flash':flash,'affinity_batch':affinity_batch},
                   'aligned':aligned,'medians':data['medians'],'path':str(result),'repeats':repeats,
                   'quality':'same frozen greedy predictions; independent holdout not yet accepted' if aligned else 'trajectory changed; revalidation required'}
            entry['runs'][key]=value;entry.pop('active',None);save()
            return value
        try:
            for label,threads,affinity in THREAD_CONFIGS:run(label,threads,affinity)
            pool=[(k,v) for k,v in entry['runs'].items() if v.get('medians') and v['repeats']==2]
            assert pool,'no successful CPU configuration'
            winner=max(pool,key=lambda item:item[1]['medians']['decode_tokens_per_second'])[1]['config']
            for label,tb,ub,fa in [('prefill4',4,512,'auto'),('ub128',winner['threads'],128,'auto'),
                                   ('fa-on',winner['threads'],512,'on'),('fa-off',winner['threads'],512,'off')]:
                run(label,winner['threads'],winner['affinity'],threads_batch=tb,ubatch=ub,flash=fa)
            run('decode2-prefill6-strict',2,'6,7',threads_batch=6,affinity_batch='2-7')
            pool=[(k,v) for k,v in entry['runs'].items() if v.get('medians') and v['repeats']==2]
            # Confirm raw winners too: different predictions require new NPU
            # feedback alignment, but cannot be discarded to lower the CPU bar.
            finalists={max(pool,key=lambda item:item[1]['medians']['decode_tokens_per_second'])[0],
                       min(pool,key=lambda item:item[1]['medians']['total_seconds'])[0],
                       min(pool,key=lambda item:item[1]['medians']['engine_ttft_seconds'])[0]}
            for label in sorted(finalists):
                cfg=entry['runs'][label]['config']
                run('final-'+label,cfg['threads'],cfg['affinity'],repeats=5,
                    threads_batch=cfg['threads_batch'],ubatch=cfg['ubatch'],flash=cfg['flash'],affinity_batch=cfg.get('affinity_batch'))
            finals={k:v for k,v in entry['runs'].items() if v.get('medians') and v['repeats']==5 and k.startswith('final-')}
            assert finals
            entry['fastest_decode']=max(finals,key=lambda k:finals[k]['medians']['decode_tokens_per_second'])
            entry['fastest_total']=min(finals,key=lambda k:finals[k]['medians']['total_seconds'])
            entry['fastest_ttft']=min(finals,key=lambda k:finals[k]['medians']['engine_ttft_seconds'])
            for new in (32,128):
                fixture=next(c for c in fixtures if c['model']==case['model'] and c['history']==case['history'] and c['new_tokens']==new)
                path=cpu_result_path(a.matrix,fixture['case'],'medium')
                deadline=time.monotonic()+14400
                while not path.exists() or json.loads(path.read_text()).get('status')!='complete':
                    if not a.wait or time.monotonic()>deadline:raise RuntimeError('sibling CPU baseline not ready')
                    if json.loads((a.matrix/'state.json').read_text()).get('status')!='running':raise RuntimeError('baseline matrix stopped')
                    time.sleep(15)
                    path=cpu_result_path(a.matrix,fixture['case'],'medium')
                for label in {entry['fastest_decode'],entry['fastest_total'],entry['fastest_ttft']}:
                    cfg=finals[label]['config']
                    run(label,cfg['threads'],cfg['affinity'],repeats=5,threads_batch=cfg['threads_batch'],
                        ubatch=cfg['ubatch'],flash=cfg['flash'],fixture=fixture,affinity_batch=cfg.get('affinity_batch'))
            entry['tested_new_tokens']=[32,64,128]
            entry.pop('confirmed_new_tokens',None)
            entry['requires_trajectory_revalidation']=[k for k,v in entry['runs'].items()
                if v.get('medians') and v['repeats']==5 and not v['aligned']]
            entry['independent_quality_accepted']=False
            entry['search_revision']=3
            entry['status']='complete';entry['finished_at']=time.time();save()
        except BaseException as exc:
            entry.update(status='failed',error=str(exc));state['status']='failed';save();raise
    state['status']='complete-selected-cases';save()


if __name__=='__main__':main()
