"""Compile full low-bit candidate; model execution and quality remain separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--bits',type=int,choices=[4,8],default=4)
p.add_argument('--variant',default='w4a16-stable')
p.add_argument('--reuse-calibration',action='store_true')
p.add_argument('--float-nonlinear',action='store_true',help='FP16 outside calibrated linear boundaries; quality diagnostic')
p.add_argument('--float-sensitive',action='store_true')
p.add_argument('--float-residual',action='store_true')
p.add_argument('--value-dfp',action='store_true')
p.add_argument('--value-bits',type=int,choices=[8])
p.add_argument('--decoder-weight-mode',choices=['perchannel','dfp'],default='perchannel')
a=p.parse_args();source=a.source.resolve();scripts=Path(__file__).resolve().parent
m=json.loads((source/'trained-manifest.json').read_text());n=m['language_layers_total']
assert len(m['layers'])==n
out=source/a.variant;out.mkdir(exist_ok=False)
def compile_one(i):
    base=source/'int16'/f'l{i}'; target=out/f'l{i}'
    with (out/f'layer{i}.log').open('x') as log:
        cmds=[
            [sys.executable,str(scripts/'compile_probe.py'),'--source',str(source),'--output',str(base),'--layer',str(i),'--dtype','int16'],
            [sys.executable,str(scripts/'compile_weight_int8.py'),'--base',str(base),'--output',str(target),'--weight-bits',str(a.bits),'--expected-linears',str(7 if i<n else 1)]]
        if a.reuse_calibration:
            state=json.loads((base/'conversion.json').read_text())
            assert state['status']=='COMPILED_NOT_BOARD_VALIDATED' and state['layer']==i
            assert (base/'decoder.quantize').exists()
            cmds=cmds[1:]
        if a.float_nonlinear:cmds[-1].append('--float-nonlinear')
        if i<n:
            cmds[-1]+=['--weight-mode',a.decoder_weight_mode]
            if a.value_bits:cmds[-1]+=['--value-bits',str(a.value_bits)]
            for flag in ('float_sensitive','float_residual','value_dfp'):
                if getattr(a,flag):cmds[-1].append('--'+flag.replace('_','-'))
        # Safe normalization graph, but retain canonical fixed INT16 IO so
        # the same runtime can load this candidate without mixed-IO guesses.
        for cmd in cmds:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            if r.returncode:
                result={'layer':i,'exit_code':r.returncode,'failed_command':cmd}
                print(json.dumps(result),flush=True);return result
    binary=target/'export_nbg_unify/network_binary.nb'
    result={'layer':i,'exit_code':0,'path':str(binary),'bytes':binary.stat().st_size,
            'sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
            'weight_bits_requested':a.bits,'float_nonlinear':a.float_nonlinear,
            'float_sensitive':a.float_sensitive if i<n else False,
            'float_residual':a.float_residual if i<n else False,
            'value_dfp':a.value_dfp if i<n else False,
            'value_bits':a.value_bits if i<n else None,
            'decoder_weight_mode':a.decoder_weight_mode if i<n else None,
            'precision':f'W{a.bits} '+(a.decoder_weight_mode if i<n else 'perchannel')+' / A16'+(' with FP16 nonlinear/IO' if a.float_nonlinear else '')+'; execution type requires device audit'}
    print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(compile_one,range(n+len(m['head_shards']))))
(out/'stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(any(r['exit_code'] for r in records))
