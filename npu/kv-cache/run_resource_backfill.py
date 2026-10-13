"""Remeasure only completed cells that predate direct response accounting.

Keep original timings immutable. The backfill has its own paths/state and must
not be silently spliced into an old run's speed/resource tuple.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from attempt_paths import select_attempt


def read(path):
    return json.loads(path.read_text())


def complete_resources(data):
    runs=[r for r in data.get('runs',[]) if r['phase']=='measure']
    return data.get('status')=='complete' and len(runs)==5 and all(r.get('resource_summary',{}).get('intervals') for r in runs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent;root=scripts.parent/'results'
    matrix=read(a.matrix/'state.json')
    state_path=a.output/'state.json'
    state=read(state_path) if state_path.exists() else {'scope':'separate paired five-run resource backfill; original timings retained', 'cases':{}}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    pending=[]
    for case,entry in matrix['cases'].items():
        for band in ('low','medium'):
            row=entry['bands'].get(band,{})
            if row.get('status')!='complete':continue
            cp=Path(row['cpu_path'])/'results.json'
            np=Path(row.get('performance',a.matrix/(case+'-'+band+'-npu.json')))
            if complete_resources(read(cp)) and complete_resources(read(np)):continue
            pending.append((case,band,entry))
    state.update(status='running',scheduled_pairs=[c+'-'+b for c,b,_ in pending]);save()
    for case,band,entry in pending:
        key=case+'-'+band
        if state['cases'].get(key,{}).get('status')=='complete':continue
        cfg=entry['config'];reference=Path(entry['reference']);gate=read(reference/'full-validation-w8.json')
        assert gate['execution_gate_v1']['passed']
        source=root/('full-smol500-stable-v3-c128' if cfg['model']=='smol500' else 'full-intern1b-sharded-v2-c128')
        group=16 if cfg['model']=='smol500' else 12
        lo,hi=matrix['bands_mC'][band]
        cpu=a.output/(key+'-cpu');npu=a.output/(key+'-npu.json')
        cpu,cpulog,cpu_attempt=select_attempt(cpu,'results.json',a.output/(key+'-cpu.log'),a.retry_failed)
        npu,npulog,npu_attempt=select_attempt(npu,None,a.output/(key+'-npu.log'),a.retry_failed)
        cpu_suffix='' if cpu_attempt==1 else f'-attempt{cpu_attempt}'
        npu_suffix='' if npu_attempt==1 else f'-attempt{npu_attempt}'
        commands=[('cpu',cpu/'results.json',[
            'run_cpu_reference.py','--prompt-folder',cfg['folder'],'--model-id',cfg['model_id'],
            '--context',str(cfg['capacity']),'--tag','resources-'+key+'-cpu'+cpu_suffix,'--output',str(cpu),
            '--repeats','5','--temperature-band-mc',str(lo),str(hi)]),
            ('npu',npu,['benchmark_variants.py','--source',str(source),'--reference',str(reference),
            '--variant','batch32',entry['tag'],reference.name.removeprefix('board-'),str(group),'2',
            '--variant-reference','batch32',str(reference),'--prefill','batch32','32','--fast-greedy','batch32',
            '--fixed-feedback','--cpu-affinity','6,7','--max-start-temp-mc','68000',
            '--temperature-band-mc',str(lo),str(hi),'--repeats','5','--experiment','resources-'+key+npu_suffix,'--output',str(npu)])]
        row=state['cases'].setdefault(key,{})
        try:
            for backend,path,command in commands:
                row.update(status='running',active=backend);save();print(key,backend,flush=True)
                if not path.exists():
                    with (cpulog if backend=='cpu' else npulog).open('x') as log:
                        subprocess.run([sys.executable,str(scripts/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
                data=read(path);assert complete_resources(data),'incomplete direct sample coverage'
                row[backend]=str(path)
                measured=[r for r in data['runs'] if r['phase']=='measure']
                actual=[r['tokens'] if backend=='cpu' else r['result']['feedback_tokens'] for r in measured]
                assert all(v==gate['runtime']['feedback_tokens'] for v in actual),'frozen trajectory mismatch'
            row.update(status='complete',reference=str(reference));row.pop('active',None);save()
        except BaseException as exc:
            row.update(status='failed',error=str(exc));state['status']='failed';save();raise
    state['status']='complete';save()


if __name__=='__main__':
    main()
