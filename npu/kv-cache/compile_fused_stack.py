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
p.add_argument('--capacity',type=int)
p.add_argument('--tokens',type=int,default=1)
p.add_argument('--value-output-fp16',action='store_true')
p.add_argument('--value-perchannel',action='store_true')
p.add_argument('--value-layout-sliced',action='store_true')
p.add_argument('--native-rmsnorm',action='store_true')
p.add_argument('--key-cache-transposed',action='store_true')
p.add_argument('--attention-fp16',action='store_true')
p.add_argument('--attention-fp16-convert',action='store_true')
p.add_argument('--reuse-blocks-from', help='reuse immutable blocks only when every source hash and graph option matches')
a = p.parse_args()
source = a.source.resolve()
out = source / a.variant
out.mkdir(exist_ok=True)
m = json.loads((source / 'trained-manifest.json').read_text())
count = len(m['layers'])
assert count % a.width == 0
base_records = json.loads((source / a.base / 'stack-conversion.json').read_text())
def compatible(record, first):
    if record.get('fusion_width') != a.width or record.get('first_layer') != first:
        return False
    if record.get('compact_gqa', False) != a.compact_gqa:
        return False
    if record.get('capacity',m['capacity'])!=(a.capacity or m['capacity']) or record.get('query_tokens',1)!=a.tokens:
        return False
    if record.get('value_output_fp16',False)!=a.value_output_fp16 or record.get('value_perchannel',False)!=a.value_perchannel:
        return False
    if record.get('value_layout_fp16',False) or record.get('value_layout_sliced',False)!=a.value_layout_sliced:
        return False
    if record.get('native_rmsnorm',False)!=a.native_rmsnorm:
        return False
    if record.get('key_cache_transposed',False)!=a.key_cache_transposed:
        return False
    if record.get('attention_fp16',False)!=a.attention_fp16:
        return False
    if record.get('attention_fp16_convert',False)!=a.attention_fp16_convert:
        return False
    if {s['layer'] for s in record['sources']} != set(range(first,first+a.width)):
        return False
    for old in record['sources']:
        if set(old['sha256']) != {'decoder.json','decoder.data','decoder.quantize'}:
            return False
        for name,digest in old['sha256'].items():
            with (source / a.base / f'l{old["layer"]}' / name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != digest:
                    return False
    return record['exit_code'] == 0 and bool(record['binaries'])
def compile_block(first):
    folder = out / f'l{first}'
    if not folder.exists() and a.reuse_blocks_from:
        target=source / a.reuse_blocks_from / f'l{first}'
        prior=json.loads((target / 'conversion.json').read_text())
        if compatible(prior,first):
            verify_contract(target,a.width)
            folder.symlink_to(target.resolve(),target_is_directory=True)
    if (folder / 'conversion.json').exists():
        record = json.loads((folder / 'conversion.json').read_text())
        assert compatible(record,first), 'existing block sources/options changed'
    else:
        with (out / f'layer{first}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('compile_fused_decoder.py')),
                '--source', str(source), '--variant', a.base, '--first', str(first), '--width', str(a.width),
                '--output', str(folder), '--tokens',str(a.tokens)] +
                (['--capacity',str(a.capacity)] if a.capacity else []) +
                (['--value-output-fp16'] if a.value_output_fp16 else []) +
                (['--value-perchannel'] if a.value_perchannel else []) +
                (['--value-layout-sliced'] if a.value_layout_sliced else []) +
                (['--native-rmsnorm'] if a.native_rmsnorm else []) +
                (['--key-cache-transposed'] if a.key_cache_transposed else []) +
                (['--attention-fp16'] if a.attention_fp16 else []) +
                (['--attention-fp16-convert'] if a.attention_fp16_convert else []) +
                (['--compact-gqa'] if a.compact_gqa else []),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=340)
        record = json.loads((folder / 'conversion.json').read_text())
    assert record['exit_code'] == 0 and record['binaries']
    verify_contract(folder, a.width)
    result = {'layer': first, 'fusion_width': a.width, 'exit_code': 0,
              'capacity':a.capacity or m['capacity'],'query_tokens':a.tokens,
              'key_cache_transposed':a.key_cache_transposed,
              'path': str(folder / 'export_nbg_unify/network_binary.nb'), 'base': a.base,
              'reused_from':str(folder.resolve()) if folder.is_symlink() else None}
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
