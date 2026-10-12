"""Compile all fused decoder blocks, retaining the existing output heads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from fused_contract import verify_contract

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--base', required=True)
p.add_argument('--variant', required=True)
p.add_argument('--width', type=int, choices=(2, 4), default=2)
p.add_argument('--jobs', type=int, choices=(1, 2), default=2)
p.add_argument('--compact-gqa', action='store_true')
a = p.parse_args()
source = a.source.resolve()
out = source / a.variant
out.mkdir(exist_ok=True)
m = json.loads((source / 'trained-manifest.json').read_text())
count = len(m['layers'])
assert count % a.width == 0
base_records = json.loads((source / a.base / 'stack-conversion.json').read_text())
def compile_block(first):
    folder = out / f'l{first}'
    if (folder / 'conversion.json').exists():
        record = json.loads((folder / 'conversion.json').read_text())
        assert record['fusion_width'] == a.width and record['first_layer'] == first
        assert record.get('compact_gqa', False) == a.compact_gqa
        for old in record['sources']:
            for name, digest in old['sha256'].items():
                assert hashlib.sha256((source / a.base / f'l{old["layer"]}' / name).read_bytes()).hexdigest() == digest
    else:
        with (out / f'layer{first}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('compile_fused_decoder.py')),
                '--source', str(source), '--variant', a.base, '--first', str(first), '--width', str(a.width),
                '--output', str(folder)] + (['--compact-gqa'] if a.compact_gqa else []),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=340)
        record = json.loads((folder / 'conversion.json').read_text())
    assert record['exit_code'] == 0 and record['binaries']
    record.update(verify_contract(folder, a.width))
    (folder / 'conversion.json').write_text(json.dumps(record, indent=2) + '\n')
    result = {'layer': first, 'fusion_width': a.width, 'exit_code': 0,
              'path': str(folder / 'export_nbg_unify/network_binary.nb'), 'base': a.base}
    print(json.dumps(result), flush=True)
    return result
with ThreadPoolExecutor(max_workers=a.jobs) as pool:
    records = list(pool.map(compile_block, range(0, count, a.width)))
for r in base_records:
    if r['layer'] < count:
        continue
    folder, target = out / f'l{r["layer"]}', source / a.base / f'l{r["layer"]}'
    if not folder.is_symlink():
        folder.symlink_to(target, target_is_directory=True)
    assert folder.resolve() == target.resolve()
    records.append({**r, 'path': str(folder / 'export_nbg_unify/network_binary.nb'), 'fusion_width': 1})
(out / 'stack-conversion.json').write_text(json.dumps(records, indent=2) + '\n')
