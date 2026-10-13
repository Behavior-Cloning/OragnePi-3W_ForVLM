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
p.add_argument('--split-kv-attention',action='store_true')
p.add_argument('--split-kv-side',choices=('both','key','value'),default='both')
p.add_argument('--split-probability-i16',action='store_true')
p.add_argument('--split-value-heads',action='store_true')
p.add_argument('--split-value-fp16',action='store_true')
p.add_argument('--split-value-transposed',action='store_true')
p.add_argument('--split-probability-identity',action='store_true')
p.add_argument('--split-softmax-i16',action='store_true')
p.add_argument('--split-current-matmul',action='store_true')
p.add_argument('--node-trace',action='store_true')
p.add_argument('--fp16-kv-cache',action='store_true')
p.add_argument('--attention-probability-i16',action='store_true')
p.add_argument('--sdk-option',action='append',default=[])
p.add_argument('--fc-input-int8',choices=('gateup','all-w8'))
p.add_argument('--fc-input-range-bits',type=int,choices=(0,1),default=0)
p.add_argument('--fc-weight-dfp',choices=('gateup','mlp','all-w8'))
p.add_argument('--fc-weight-dfp-method',choices=('peak','mse'),default='peak')
p.add_argument('--fc-weight-dfp-max-extra-bits',type=int,choices=range(7),default=6)
p.add_argument('--mlp-product-headroom',type=int,nargs=2,action='append',default=[],metavar=('LAYER','BITS'))
p.add_argument('--query-headroom',type=int,nargs=2,action='append',default=[],metavar=('LAYER','BITS'))
p.add_argument('--reuse-blocks-from', help='reuse immutable blocks only when every source hash and graph option matches')
a = p.parse_args()
source = a.source.resolve()
out = source / a.variant
out.mkdir(exist_ok=True)
m = json.loads((source / 'trained-manifest.json').read_text())
count = len(m['layers'])
assert count % a.width == 0
assert len(dict(a.mlp_product_headroom))==len(a.mlp_product_headroom)
assert all(0<=layer<count and 1<=bits<=4 for layer,bits in a.mlp_product_headroom)
assert len(dict(a.query_headroom))==len(a.query_headroom)
assert all(0<=layer<count and 1<=bits<=4 for layer,bits in a.query_headroom)
base_records = json.loads((source / a.base / 'stack-conversion.json').read_text())
def compatible(record, first):
    if sorted(record.get('query_headroom',[]))!=sorted(p for p in a.query_headroom if first<=p[0]<first+a.width):
        return False
    ranges=[p for p in a.mlp_product_headroom if first<=p[0]<first+a.width]
    if sorted(record.get('mlp_product_headroom',[]))!=sorted(ranges):
        return False  # Never silently reuse a graph with different ranges.
    if record.get('fc_weight_dfp') != a.fc_weight_dfp:
        return False
    if record.get('fc_weight_dfp_method','peak') != a.fc_weight_dfp_method:
        return False
    if record.get('fc_weight_dfp_max_extra_bits',6) != a.fc_weight_dfp_max_extra_bits:
        return False
    if record.get('fc_input_int8') != a.fc_input_int8 or record.get('fc_input_range_bits',0) != a.fc_input_range_bits:
        return False
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
    if record.get('split_kv_attention',False)!=a.split_kv_attention:
        return False
    if record.get('fp16_kv_cache',False)!=a.fp16_kv_cache:
        return False
    if record.get('attention_probability_i16',False)!=a.attention_probability_i16:
        return False
    if a.split_kv_attention and (record.get('split_kv_side','both')!=a.split_kv_side or
            record.get('split_probability_i16',False)!=a.split_probability_i16 or
            record.get('split_value_heads',False)!=a.split_value_heads or
            record.get('split_value_fp16',False)!=a.split_value_fp16 or
            record.get('split_value_transposed',False)!=a.split_value_transposed or
            record.get('split_probability_identity',False)!=a.split_probability_identity or
            record.get('split_softmax_i16',False)!=a.split_softmax_i16 or
            record.get('split_current_matmul',False)!=a.split_current_matmul):
        return False
    if sorted(record.get('sdk_options',[]))!=sorted(a.sdk_option):
        return False
    if a.node_trace and not record.get('node_trace',False):
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
                (['--split-kv-attention','--split-kv-side',a.split_kv_side] if a.split_kv_attention else []) +
                (['--split-probability-i16'] if a.split_probability_i16 else []) +
                (['--split-value-heads'] if a.split_value_heads else []) +
                (['--split-value-fp16'] if a.split_value_fp16 else []) +
                (['--split-value-transposed'] if a.split_value_transposed else []) +
                (['--split-probability-identity'] if a.split_probability_identity else []) +
                (['--split-softmax-i16'] if a.split_softmax_i16 else []) +
                (['--split-current-matmul'] if a.split_current_matmul else []) +
                (['--node-trace'] if a.node_trace else []) +
                (['--fp16-kv-cache'] if a.fp16_kv_cache else []) +
                (['--attention-probability-i16'] if a.attention_probability_i16 else []) +
                [v for opt in a.sdk_option for v in ('--sdk-option',opt)] +
                (['--fc-input-int8',a.fc_input_int8,'--fc-input-range-bits',str(a.fc_input_range_bits)] if a.fc_input_int8 else []) +
                (['--fc-weight-dfp',a.fc_weight_dfp,'--fc-weight-dfp-method',a.fc_weight_dfp_method,
                  '--fc-weight-dfp-max-extra-bits',str(a.fc_weight_dfp_max_extra_bits)] if a.fc_weight_dfp else []) +
                [v for layer,bits in a.mlp_product_headroom if first<=layer<first+a.width
                   for v in ('--mlp-product-headroom',str(layer),str(bits))] +
                [v for layer,bits in a.query_headroom if first<=layer<first+a.width
                   for v in ('--query-headroom',str(layer),str(bits))] +
                (['--compact-gqa'] if a.compact_gqa else []),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=340)
        record = json.loads((folder / 'conversion.json').read_text())
    assert record['exit_code'] == 0 and record['binaries']
    verify_contract(folder, a.width)
    result = {'layer': first, 'fusion_width': a.width, 'exit_code': 0,
              'capacity':a.capacity or m['capacity'],'query_tokens':a.tokens,
              'key_cache_transposed':a.key_cache_transposed,
              'split_kv_attention':a.split_kv_attention,'split_kv_side':a.split_kv_side,
              'fp16_kv_cache':a.fp16_kv_cache,
              'attention_probability_i16':a.attention_probability_i16,
              'split_probability_i16':a.split_probability_i16,'sdk_options':a.sdk_option,
              'split_value_heads':a.split_value_heads,'split_value_fp16':a.split_value_fp16,
              'split_value_transposed':a.split_value_transposed,
              'split_probability_identity':a.split_probability_identity,
              'split_softmax_i16':a.split_softmax_i16,
              'split_current_matmul':a.split_current_matmul,
              'mlp_product_headroom':[p for p in a.mlp_product_headroom if first<=p[0]<first+a.width],
              'query_headroom':[p for p in a.query_headroom if first<=p[0]<first+a.width],
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
