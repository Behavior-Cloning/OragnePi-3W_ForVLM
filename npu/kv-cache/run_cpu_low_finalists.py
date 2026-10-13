"""Confirm medium-search CPU finalists at low start temperature, without claiming
the medium shortlist exhausts all low-temperature implementations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from attempt_paths import select_attempt


def finalists(entry, new_tokens):
    prefix='' if new_tokens==64 else f'n{new_tokens}-'
    pool={k:v for k,v in entry.get('runs',{}).items()
          if k.startswith(prefix+'final-') and v.get('repeats')==5 and v.get('medians')}
    if not pool:
        return {}
    selected={max(pool,key=lambda k:pool[k]['medians']['decode_tokens_per_second']),
              min(pool,key=lambda k:pool[k]['medians']['engine_ttft_seconds']),
              min(pool,key=lambda k:pool[k]['medians']['total_seconds'])}
    # Do not drop a faster candidate merely because its greedy trajectory
    # differs; that requires alignment/quality checks, not a slower CPU bar.
    return {k:pool[k] for k in sorted(selected)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--search',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    scripts=Path(__file__).resolve().parent
    fixtures=json.loads((a.fixtures/'matrix.json').read_text())['cases']
    a.output.mkdir(parents=True,exist_ok=True)
    state_path=a.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'status':'running','cases':{},'scope':'medium CPU shortlist remeasured at low start 40–50 C; not exhaustive low-temperature search',
        'independent_quality_accepted':False}
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    deadline=time.monotonic()+14400
    while True:
        search=json.loads((a.search/'state.json').read_text())
        for fixture in fixtures:
            case=fixture['case']
            if state['cases'].get(case,{}).get('status')=='complete':continue
            parent=case.rsplit('-n',1)[0]+'-n64'
            entry=search['cases'].get(parent,{})
            if entry.get('status')!='complete':continue
            pool=finalists(entry,fixture['new_tokens'])
            assert pool, 'completed search missing exact-length finalists'
            row=state['cases'].setdefault(case,{'status':'running','runs':{}})
            try:
                for key,source in pool.items():
                    label=key.split('-',1)[1] if key.startswith(('n32-','n128-')) else key
                    folder,log,attempt=select_attempt(a.output/(case+'-'+label),'results.json',
                        a.output/(case+'-'+label+'.log'),a.retry_failed)
                    result=folder/'results.json'
                    if not result.exists():
                        cfg=source['config']
                        command=[sys.executable,str(scripts/'run_cpu_reference.py'),
                            '--prompt-folder',fixture['folder'],'--model-id',fixture['model_id'],
                            '--context',str(fixture['capacity']),'--tag','q8low-'+case+'-'+label+
                            ('' if attempt==1 else f'-attempt{attempt}'),'--output',str(folder),
                            '--repeats','5','--threads',str(cfg['threads']),'--cpu-affinity',cfg['affinity'],
                            '--threads-batch',str(cfg['threads_batch']),'--ubatch-size',str(cfg['ubatch']),
                            '--flash-attn',cfg['flash'],'--temperature-band-mc','40000','50000',
                            '--max-start-temp-mc','50000']
                        if cfg.get('affinity_batch'):command+=['--cpu-affinity-batch',cfg['affinity_batch']]
                        row.update(status='running',active=label);state['status']='running';save()
                        with log.open('x') as stream:
                            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    data=json.loads(result.read_text())
                    measured=[r for r in data['runs'] if r['phase']=='measure']
                    assert data['status']=='complete' and len(measured)==5
                    assert data['temperature_band_mC']==[40000,50000]
                    assert all(40000<=r['response_start_temperature_mC']<=50000 for r in measured)
                    assert data['cached_prefix']==fixture['history']
                    assert len(data['prompt']['token_ids'])==fixture['history']+fixture['new_tokens']
                    previous=json.loads(Path(source['path']).read_text())
                    assert previous['prompt']==data['prompt']
                    assert all(r['tokens']==data['nonstream_trajectory']['tokens'] for r in measured)
                    row['runs'][label]={'path':str(result),'source_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                        'medium_source':source['path'],'config':source['config'],'medians':data['medians'],
                        'repeats':5,'aligned_with_medium':data['nonstream_trajectory']['tokens']==previous['nonstream_trajectory']['tokens']}
                    save()
                row.update(status='complete',active=None)
                row['fastest_decode']=max(row['runs'],key=lambda k:row['runs'][k]['medians']['decode_tokens_per_second'])
                row['fastest_total']=min(row['runs'],key=lambda k:row['runs'][k]['medians']['total_seconds'])
                row['fastest_ttft']=min(row['runs'],key=lambda k:row['runs'][k]['medians']['engine_ttft_seconds'])
                save()
            except BaseException as exc:
                row.update(status='failed',error=str(exc));state['status']='failed';save();raise
        if all(state['cases'].get(f['case'],{}).get('status')=='complete' for f in fixtures):break
        assert search['status']!='failed', 'CPU search failed; preserve evidence and inspect before continuing'
        assert time.monotonic()<deadline, 'CPU finalists not available within four hours'
        state['status']='waiting-medium-search';save();time.sleep(15)
    state['status']='complete';save()


if __name__=='__main__':main()
