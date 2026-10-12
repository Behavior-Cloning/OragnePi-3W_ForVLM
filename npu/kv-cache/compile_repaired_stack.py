"""Compile full INT8 linear-weight stack, with selectable sensitive precision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--dtype',default='w8-vdfp');p.add_argument('--wait-base',type=int,default=900)
p.add_argument('--float-residual',action='store_true')
p.add_argument('--float-sensitive',action='store_true')
p.add_argument('--activation-headroom-bits',type=int,default=0)
p.add_argument('--fp16-head',action='store_true')
args=p.parse_args();source=args.source;manifest=json.loads((source/'trained-manifest.json').read_text())
out=source/args.dtype;out.mkdir(exist_ok=True)
def compile_one(index):
    base=source/'int16'/f'l{index}';deadline=time.monotonic()+args.wait_base
    while True:
        path=base/'conversion.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('status')=='COMPILED_NOT_BOARD_VALIDATED':break
            if any(stage['exit'] for stage in state['stages']):raise RuntimeError(f'base layer {index} failed')
        if time.monotonic()>deadline:raise TimeoutError(f'base layer {index}')
        time.sleep(2)
    target=out/f'l{index}'
    if index==len(manifest['layers']):
        (target/'export_nbg_unify').mkdir(parents=True,exist_ok=True)
        head_base=source/'fp16'/f'l{index}' if args.fp16_head else base
        shutil.copyfile(head_base/'export_nbg_unify/network_binary.nb',target/'export_nbg_unify/network_binary.nb')
        result=0
    else:
        with (out/f'layer{index}.log').open('w') as log:
            command=[sys.executable,str(Path(__file__).with_name('compile_weight_int8.py')),
                '--base',str(base),'--output',str(target),'--value-dfp']
            if args.float_residual:command.append('--float-residual')
            if args.float_sensitive:command.append('--float-sensitive')
            command+=['--activation-headroom-bits',str(args.activation_headroom_bits)]
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
    record={'layer':index,'exit_code':result,'path':str(target/'export_nbg_unify/network_binary.nb'),
            'precision':('FP16 LM head' if args.fp16_head else 'INT16 LM head') if index==len(manifest['layers']) else 'W8 (V DFP, other linears per-channel), A16',
            'float_residual':args.float_residual,'float_sensitive':args.float_sensitive}
    print(json.dumps(record),flush=True);return record
with ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(compile_one,range(len(manifest['layers'])+1)))
(out/'stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(any(r['exit_code'] for r in records))
