"""Expand a gated 1B prefill candidate with paired full-depth five-run A/B."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent;project=scripts.parents[1]
    source=project/'npu/results/full-intern1b-sharded-v2-c128'
    dtype='w8-fused2-c768-splitkv-i16mm';tag='full-intern1b-c768-prefill16'
    model=project/'models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf'
    matrix=json.loads((a.matrix/'state.json').read_text())
    statepath=a.output/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'status':'running','cases':{}}
    def save():
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath)
    def run(name,step,command):
        with (a.output/(name+'-'+step+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(scripts/command[0]),*map(str,command[1:])],stdout=log,stderr=subprocess.STDOUT,check=True)
    for new in (32,128):
        name=f'intern1b-h512-n{new}';case='prefill16-'+name
        if state['cases'].get(name,{}).get('status')=='complete':continue
        entry=matrix['cases'][name];reference=Path(entry['reference'])
        assert entry['bands']['medium']['status']=='complete'
        gate=json.loads((reference/'full-validation-w8.json').read_text());assert gate['execution_gate_v1']['passed']
        candidate=source/dtype/('board-'+case)
        row=state['cases'].setdefault(name,{});row.update(status='running',active='full-depth-validation');save()
        try:
            if not (candidate/'tokens.json').exists():
                run(name,'full',['run_full_model.py','--source',source,'--dtype',dtype,'--tag',tag,'--case',case,
                    '--prompt-folder',entry['config']['folder'],'--max-new','128','--cached-prefix','512',
                    '--decoder-group','12','--fusion-width','2','--prefill-variant','w8-fused2-c768-t16-splitkv-i16mm',
                    '--fast-greedy','--group-head','--cpu-affinity','6,7','--skip-deploy',
                    '--feedback-from',Path(entry['bands']['low']['cpu_path'])/'tokens.json'])
            for weights,suffix in [('exported','-w8'),('original','')]:
                if not (candidate/('full-validation'+suffix+'.json')).exists():
                    run(name,weights,['validate_full.py','--model',model,'--source',source,'--dtype',dtype,'--case',case,'--reference-weights',weights])
            validation=json.loads((candidate/'full-validation-w8.json').read_text())
            assert validation['execution_gate_v1']['passed'],'candidate fails numerical gate'
            row.update(active='five-repeat-ab',reference=str(candidate));save()
            output=a.output/(name+'-ab.json')
            if not output.exists():
                run(name,'ab',['benchmark_variants.py','--source',source,'--reference',reference,
                    '--variant','t32',entry['tag'],reference.name.removeprefix('board-'),'12','2',
                    '--variant','t16',tag,case,'12','2','--variant-reference','t32',reference,
                    '--variant-reference','t16',candidate,'--prefill','t32','32','--prefill','t16','16',
                    '--fast-greedy','t32','--fast-greedy','t16','--group-head','t32','--group-head','t16',
                    '--fixed-feedback','--cpu-affinity','6,7','--max-start-temp-mc','68000',
                    '--temperature-band-mc','55000','64999','--repeats','5','--experiment','prefill16-'+name+'-ab',
                    '--output',output])
            result=json.loads(output.read_text());assert result['status']=='complete'
            row.update(status='complete',result=str(output),medians=result['medians']);row.pop('active',None);save()
        except BaseException as exc:
            row.update(status='failed',error=str(exc));state['status']='failed';save();raise
    state['status']='complete';save()


if __name__=='__main__':main()
