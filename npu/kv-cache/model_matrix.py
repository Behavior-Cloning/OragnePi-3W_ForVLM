"""Bounded model-layer matrix, preserving failures instead of calling them passes."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--tag',required=True)
p.add_argument('--capacities',type=int,nargs='+',default=[128,512])
p.add_argument('--reuse-export',action='store_true')
args=p.parse_args()
if not args.tag.replace('-','').isalnum():raise ValueError('invalid tag')
root=Path(__file__).resolve().parents[2]
scripts=Path(__file__).resolve().parent
report={'tag':args.tag,'model':str(args.model),'scope':'two trained decoder layers, not full generation','cases':[]}
report_path=root/f'npu/results/{args.tag}-matrix.json'

def run(name,script,argv,folder,timeout=900):
    start=time.monotonic()
    with (folder/f'matrix-{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(scripts/script),*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    record={'stage':name,'exit_code':result.returncode,'seconds':time.monotonic()-start}
    print(json.dumps(record),flush=True)
    return record

for capacity in args.capacities:
    folder=root/f'npu/results/{args.tag}-c{capacity}';folder.mkdir(parents=True,exist_ok=True)
    case={'capacity':capacity,'stages':[]};report['cases'].append(case)
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
    if args.reuse_export:
        existing=json.loads((folder/'trained-manifest.json').read_text())
        assert existing['capacity']==capacity
        assert existing['sha256']==hashlib.file_digest(args.model.open('rb'),'sha256').hexdigest()
        stage={'stage':'export','exit_code':0,'reused_verified_manifest':True}
    else:
        stage=run('export','trained_probe.py',['--model',args.model,'--output',folder,'--capacity',capacity],folder)
    case['stages'].append(stage);save()
    if stage['exit_code']:continue
    manifest=json.loads((folder/'trained-manifest.json').read_text())
    for dtype in ('int16','fp16'):
        def compile_layer(layer):
            return run(f'compile-{dtype}-l{layer}','compile_probe.py',
                ['--source',folder,'--output',root/f'npu/results/{args.tag}-c{capacity}-l{layer}-{dtype}',
                 '--layer',layer,'--dtype',dtype],folder)
        with ThreadPoolExecutor(max_workers=2) as pool:
            compiled=list(pool.map(compile_layer,(0,1)))
        case['stages'].extend(compiled);save()
        if any(s['exit_code'] for s in compiled):continue
        stage=run(f'board-{dtype}','run_board_probe.py',
            ['--capacity',capacity,'--fixture',folder,
             '--layer0',root/f'npu/results/{args.tag}-c{capacity}-l0-{dtype}/export_nbg_unify/network_binary.nb',
             '--layer1',root/f'npu/results/{args.tag}-c{capacity}-l1-{dtype}/export_nbg_unify/network_binary.nb',
             '--tag',f'{args.tag}-c{capacity}-{dtype}','--dim',manifest['dim'],
             '--kv',manifest['kv_heads'],'--head-dim',manifest['head_dim'],'--rope-base',manifest['rope_base']],folder)
        case['stages'].append(stage);save()
    stage=run('cpu','run_cpu_comparison.py',['--capacity',capacity,'--folder',folder,'--tag',args.tag],folder,timeout=7200)
    case['stages'].append(stage);save()
print(json.dumps(report,indent=2))
