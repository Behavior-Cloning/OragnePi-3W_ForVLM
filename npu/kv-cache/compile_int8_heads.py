"""Quantize every vocabulary shard; retain decoder artifacts for controlled comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--base',required=True);p.add_argument('--dtype',required=True)
p.add_argument('--float-head-norm',action='store_true')
p.add_argument('--reuse-int16-base',action='store_true')
p.add_argument('--activation-headroom-bits',type=int,default=0)
p.add_argument('--weight-mode',choices=['perchannel','dfp'],default='perchannel')
p.add_argument('--dfp-method',choices=['peak','mse'],default='peak')
args=p.parse_args();scripts=Path(__file__).resolve().parent
manifest=json.loads((args.source/'trained-manifest.json').read_text());count=manifest['language_layers_total']
out=args.source/args.dtype;out.mkdir(exist_ok=True)
shards=manifest.get('head_shards',[{'layer':count}])
def compile_head(shard):
    i=shard['layer'];base=args.source/'int16-heads'/f'l{i}'
    with (out/f'head{i}.log').open('w') as log:
        commands=[
            [sys.executable,str(scripts/'compile_probe.py'),'--source',str(args.source),'--output',str(base),'--layer',str(i),'--dtype','int16'],
            [sys.executable,str(scripts/'compile_weight_int8.py'),'--base',str(base),'--output',str(out/f'l{i}'),'--expected-linears','1']]
        if args.float_head_norm:commands[-1].append('--float-head-norm')
        commands[-1]+=['--activation-headroom-bits',str(args.activation_headroom_bits)]
        commands[-1]+=['--weight-mode',args.weight_mode,'--dfp-method',args.dfp_method]
        if args.reuse_int16_base:
            state=json.loads((base/'conversion.json').read_text())
            if state.get('status')!='COMPILED_NOT_BOARD_VALIDATED':raise ValueError('invalid calibrated head base')
            commands=commands[1:]
        for command in commands:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    print(json.dumps({'head':i,'status':'compiled, board validation pending'}),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile_head,shards))
subprocess.run([sys.executable,str(scripts/'assemble_stack_variant.py'),'--source',str(args.source),
    '--base',args.base,'--dtype',args.dtype,'--replace',*[str(s['layer']) for s in shards]],check=True)
