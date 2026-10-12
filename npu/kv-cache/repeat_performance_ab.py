"""Alternate two complete NPU chains, checking token equality on every clean run."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
p.add_argument('--baseline-tag',required=True);p.add_argument('--candidate-tag',required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=5);p.add_argument('--max-new',type=int,default=96)
p.add_argument('--case-prefix',default='perf-ab')
args=p.parse_args();script=Path(__file__).with_name('run_full_model.py')
if not 1<=args.repeats<=20:raise ValueError('invalid repeats')
expected=json.loads((args.reference/'tokens.json').read_text())['tokens']
report={'scope':'same prompt and verified identical natural-EOS trajectory; full chain, no logits/trace writes',
    'baseline':args.baseline,'candidate':args.candidate,'runs':[],'status':'running'}
args.output.parent.mkdir(parents=True,exist_ok=True)
for repeat in range(args.repeats+1):
    order=[(args.baseline,args.baseline_tag),(args.candidate,args.candidate_tag)]
    if repeat%2:order.reverse()
    for dtype,tag in order:
        case=f'{args.case_prefix}-{repeat}'
        with (args.source/dtype/(case+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(script),'--source',str(args.source),'--dtype',dtype,
                '--tag',tag,'--prompt-folder',str(args.reference),'--case',case,'--max-new',str(args.max_new),
                '--skip-deploy','--performance'],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.source/dtype/('board-'+case)/'tokens.json').read_text())
        if result['tokens']!=expected:raise ValueError('token trajectory changed')
        if result['driver_after_release_bytes']!=0:raise ValueError('NPU allocation not released')
        if result['temperature_before_mC']>55000:raise ValueError('starting temperature above 55C; discard this comparison')
        report['runs'].append({'dtype':dtype,'repeat':repeat,'warmup':repeat==0,'result':result})
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(dtype,repeat,result['decode_tokens_per_second'],flush=True)
report['medians']={dtype:{key:statistics.median(r['result'][key] for r in report['runs']
    if r['dtype']==dtype and not r['warmup']) for key in
    ('decode_tokens_per_second','ttft_seconds','total_seconds','cpu_seconds','max_rss_kib','driver_allocated_bytes')}
    for dtype in (args.baseline,args.candidate)}
report['speedup']=report['medians'][args.candidate]['decode_tokens_per_second']/report['medians'][args.baseline]['decode_tokens_per_second']
report['status']='complete'
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['medians'],indent=2))
