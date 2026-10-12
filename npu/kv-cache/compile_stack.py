"""Compile every language decoder layer and the final head, retaining stage logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--dtype',choices=['fp16','int16','int8'],default='fp16');args=p.parse_args()
manifest=json.loads((args.source/'trained-manifest.json').read_text())
assert len(manifest['layers'])==manifest['language_layers_total']
out=args.source/args.dtype;out.mkdir(exist_ok=True)
def compile_one(index):
    folder=out/f'l{index}'
    with (out/f'layer{index}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('compile_probe.py')),
            '--source',str(args.source),'--output',str(folder),'--layer',str(index),'--dtype',args.dtype],
            stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    record={'layer':index,'exit_code':result.returncode,'path':str(folder/'export_nbg_unify/network_binary.nb')}
    print(json.dumps(record),flush=True);return record
with ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(compile_one,range(len(manifest['layers'])+1)))
(out/'stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(any(r['exit_code'] for r in records))
