"""Repeat the repaired full-model development case; retain every raw result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--model',type=Path,required=True);p.add_argument('--dtype',required=True)
p.add_argument('--tag',required=True);p.add_argument('--prompt-folder',type=Path,required=True)
p.add_argument('--repeats',type=int,default=4)
args=p.parse_args();scripts=Path(__file__).resolve().parent
records=[]
for i in range(2,args.repeats+2):
    case=f'companion-repeat{i}'
    with (args.source/args.dtype/(case+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(scripts/'run_full_model.py'),'--source',str(args.source),
            '--dtype',args.dtype,'--tag',args.tag,'--prompt-folder',str(args.prompt_folder),
            '--case',case,'--max-new','96','--skip-deploy'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(scripts/'validate_full.py'),'--source',str(args.source),
            '--model',str(args.model),'--dtype',args.dtype,'--case',case,'--reference-weights','exported'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((args.source/args.dtype/('board-'+case)/'full-validation-w8.json').read_text())
    records.append({'case':case,'gate':result['execution_gate_v1'],'runtime':result['runtime'],
                    'relative_l2':result['relative_l2_all_logits'],'answer':result['npu_answer']})
    (args.source/args.dtype/'repeat-validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print(case,result['execution_gate_v1']['passed'],result['relative_l2_all_logits'],flush=True)
    if not result['execution_gate_v1']['passed']:raise SystemExit('repeat numerical regression')
