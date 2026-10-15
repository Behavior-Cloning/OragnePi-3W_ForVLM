"""Serial full-depth submission screening; rejects changed output trajectories."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--groups',type=int,nargs='+',default=[0,1,2,4,7,14,28]);p.add_argument('--label',required=True);a=p.parse_args()
assert a.label.replace('-','').isalnum()
root=Path(__file__).resolve().parents[2];scripts=root/'npu/kv-cache'
source=root/'npu/results/full-qwen25-15b-sharded-min-c128';variant='w4-vdfp-sensitive'
reference=json.loads((source/variant/'board-clean-group4/tokens.json').read_text())
out=root/'npu/results'/a.label;out.mkdir(exist_ok=False);rows=[]
for group in a.groups:
    assert 0<=group<=28
    case=f'{a.label}-g{group}'
    cmd=[sys.executable,str(scripts/'run_full_model.py'),'--source',str(source),
         '--prompt-folder',str(root/'npu/results/full-qwen25-15b-sharded-c128'),
         '--dtype',variant,'--tag','full-qwen15-vdfp-c128','--skip-deploy',
         '--case',case,'--max-new','64','--performance','--group-head','--fast-greedy',
         '--cpu-affinity','6,7','--decoder-group',str(group)]
    with (out/f'g{group}.log').open('x') as log:
        r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    row={'group':group,'exit_code':r.returncode}
    if r.returncode==0:
        result=json.loads((source/variant/('board-'+case)/'tokens.json').read_text())
        row.update(runtime=result,tokens_identical=result['tokens']==reference['tokens'])
    rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='runtime'}),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
    if not row['tokens_identical']:raise RuntimeError('changed trajectory; stop performance search')
