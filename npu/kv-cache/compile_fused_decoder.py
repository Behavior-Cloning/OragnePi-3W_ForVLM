"""Join calibrated ACUITY decoder graphs, with explicit optional experiments.

Only the intermediate hidden input/output boundary is removed. Every layer
keeps its own KV, mask and RoPE ports, so no scale compatibility is assumed.
Layout and precision experiments are opt-in and recorded in conversion.json.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import yaml
from fused_contract import verify_contract

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--variant', required=True)
p.add_argument('--first', type=int, required=True)
p.add_argument('--width', type=int, choices=(2, 4), default=2)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--timeout', type=int, default=300)
p.add_argument('--compact-gqa', action='store_true')
p.add_argument('--capacity',type=int,help='resize KV capacity and retain calibrated tensor scales')
p.add_argument('--tokens',type=int,default=1,help='query token count for a causal prefill block')
p.add_argument('--value-output-fp16',action='store_true',help='diagnostic explicit FP16 conversion at external V output only')
p.add_argument('--value-perchannel',action='store_true',help='diagnostic W8 per-channel V kernel for batched queries')
p.add_argument('--value-layout-fp16',action='store_true',help='diagnostic FP16 V layout conversion with unchanged matrix weights')
p.add_argument('--value-layout-sliced',action='store_true',help='replace V permute by equivalent per-KV-head slices')
p.add_argument('--native-rmsnorm',action='store_true',help='experimental SDK RMSNormalize; requires independent numerical validation')
p.add_argument('--key-cache-transposed',action='store_true',help='experimental K cache layout; requires matching runtime')
p.add_argument('--attention-fp16',action='store_true',help='diagnostic FP16 matmul operand conversion; KV ports stay calibrated')
p.add_argument('--attention-fp16-convert',action='store_true',help='explicit dtype_converter before attention matmul; preserve layout-node precision')
p.add_argument('--node-trace',action='store_true',help='emit SDK node mapping for NPD diagnostics')
p.add_argument('--fp16-kv-cache',action='store_true',help='retain history in FP16 and convert only newly appended KV')
p.add_argument('--attention-probability-i16',action='store_true',help='ablation: INT16 softmax probability output with original KV concat')
p.add_argument('--split-kv-attention',action='store_true',help='avoid full-history KV concat by distributed attention matmuls')
p.add_argument('--split-kv-side',choices=('both','key','value'),default='both')
p.add_argument('--split-probability-i16',action='store_true',help='materialize sliced past probabilities in INT16')
p.add_argument('--split-value-heads',action='store_true',help='lower past AV to per-KV-head 2D matrices')
p.add_argument('--split-value-fp16',action='store_true',help='explicitly convert past V before split attention')
p.add_argument('--split-value-transposed',action='store_true',help='evaluate AV as transposed V times transposed probabilities')
p.add_argument('--split-probability-identity',action='store_true',help='materialize positive sliced probabilities through abs')
p.add_argument('--split-softmax-i16',action='store_true',help='retain softmax arithmetic but quantize its probabilities to INT16 at the producer')
p.add_argument('--split-current-matmul',action='store_true',help='use outer-product matmul for current V instead of cross-axis broadcasting')
p.add_argument('--sdk-option',action='append',default=[],metavar='VIV_VX_NAME=VALUE',help='explicit SDK experiment, recorded in provenance')
p.add_argument('--fc-input-int8',choices=('gateup','all-w8'),help='diagnostic local A8 converters for selected W8 matrices; preserve KV and outputs')
p.add_argument('--fc-input-range-bits',type=int,choices=(0,1),default=0)
p.add_argument('--fc-weight-dfp',choices=('gateup','all-w8'),help='diagnostic W8 single-scale weight encoding; separate from A8 trials')
p.add_argument('--fc-weight-dfp-method',choices=('peak','mse'),default='peak')
p.add_argument('--mlp-product-headroom',type=int,nargs=2,action='append',default=[],metavar=('LAYER','BITS'))
a = p.parse_args()
product_headroom=dict(a.mlp_product_headroom)
assert len(product_headroom)==len(a.mlp_product_headroom)
assert all(a.first<=layer<a.first+a.width and 1<=bits<=4 for layer,bits in product_headroom.items())
assert a.fc_weight_dfp or a.fc_weight_dfp_method=='peak'
assert not (a.fc_input_int8 and a.fc_weight_dfp), 'isolate activation and weight encoding trials'
assert not (a.attention_fp16 and a.attention_fp16_convert)
assert not (a.split_kv_side != 'both' or any((a.split_probability_i16, a.split_value_heads,
    a.split_value_fp16, a.split_value_transposed, a.split_probability_identity,
    a.split_softmax_i16, a.split_current_matmul))) or a.split_kv_attention
assert not a.split_kv_attention or a.tokens == 1 or a.split_current_matmul
assert all(re.fullmatch(r'VIV_VX_[A-Z0-9_]+=[A-Za-z0-9_.-]+', v) for v in a.sdk_option)
assert len({v.split('=')[0] for v in a.sdk_option}) == len(a.sdk_option)
root = Path(__file__).resolve().parents[2] / 'npu/results'
out = a.output.resolve()
relative = out.relative_to(root)
out.mkdir(parents=True, exist_ok=False)
nodes, weights, quant, ports, output_nodes = {}, {}, {}, [], []
sources = []
layout_changes = []
norm_changes = []
key_layout_changes = []
attention_changes = []
split_kv_changes = []
fp16_kv_changes = []
fc_input_changes = []
fc_weight_changes = []
product_range_changes = []
manifest = json.loads((a.source / 'trained-manifest.json').read_text())
previous_hidden = None
for offset in range(a.width):
    folder = a.source / a.variant / f'l{a.first + offset}'
    graph = json.loads((folder / 'decoder.json').read_text())
    q = yaml.safe_load((folder / 'decoder.quantize').read_text())
    data = np.load(folder / 'decoder.data', allow_pickle=True).item()
    if a.first+offset in product_headroom:
        from activation_ranges import widen_mlp_product
        product_range_changes.append({'layer':a.first+offset,**widen_mlp_product(graph,q,product_headroom[a.first+offset])})
    if a.fc_weight_dfp:
        from fc_input_precision import convert_fc_weights_dfp
        fc_weight_changes.append({'layer':a.first+offset,**convert_fc_weights_dfp(
            graph,q,data,manifest['dim'],a.fc_weight_dfp,a.fc_weight_dfp_method)})
    if a.fc_input_int8:
        from fc_input_precision import convert_fc_inputs
        fc_input_changes.append({'layer':a.first+offset,**convert_fc_inputs(
            graph,q,data,manifest['dim'],a.fc_input_int8,a.fc_input_range_bits)})
    if a.value_output_fp16:
        for key,node in graph['Layers'].items():
            if node['name']=='attach_new_v/out0':
                q['quantize_parameters']['@'+key+':out0']={'qtype':'float16','quantizer':'float16'}
    if a.value_perchannel:
        original_nodes=graph['Layers']
        key=next(k for k,v in original_nodes.items() if v['name']=='new_v')
        while original_nodes[key]['op']!='fullconnect':
            assert len(original_nodes[key]['inputs'])==1
            key=original_nodes[key]['inputs'][0][1:].rsplit(':',1)[0]
        weight=data[key]['weight']
        scales=np.maximum(np.abs(weight).max(axis=0)/127,np.finfo(np.float32).tiny)
        q['quantize_parameters']['@'+key+':weight']={'qtype':'i8','quantizer':'perchannel_symmetric_affine',
            'rounding':'rtne','scale':scales.tolist(),'channel_dim':-1}
    if a.value_layout_fp16:
        original_nodes=graph['Layers']
        key=next(k for k,v in original_nodes.items() if v['name']=='new_v')
        while original_nodes[key]['op']!='fullconnect':
            q['quantize_parameters']['@'+key+':out0']={'qtype':'float16','quantizer':'float16'}
            assert len(original_nodes[key]['inputs'])==1
            key=original_nodes[key]['inputs'][0][1:].rsplit(':',1)[0]
    if a.capacity is not None or a.tokens!=1:
        assert a.compact_gqa, 'resizing requires compact GQA'
        from reshape_decoder import reshape_decoder
        data,change=reshape_decoder(graph,q,data,manifest['heads'],manifest['kv_heads'],manifest['head_dim'],
            a.capacity or manifest['capacity'],a.tokens)
        layout_changes.append({'layer':a.first+offset,**change})
    elif a.compact_gqa:
        from compact_gqa import compact_gqa
        data, change = compact_gqa(graph, q, data, manifest['heads'], manifest['kv_heads'], manifest['head_dim'])
        layout_changes.append({'layer': a.first + offset, **change})
    if a.value_layout_sliced:
        from reshape_decoder import slice_value_layout
        slice_value_layout(graph,q,manifest['kv_heads'],manifest['head_dim'],a.tokens)
    if a.fp16_kv_cache:
        assert a.compact_gqa and not any((a.key_cache_transposed,a.split_kv_attention,a.attention_fp16,a.attention_fp16_convert))
        from fp16_kv_cache import fp16_kv_cache
        fp16_kv_changes.append({'layer':a.first+offset,**fp16_kv_cache(graph,q)})
    if a.attention_probability_i16:
        assert a.compact_gqa and not any((a.split_kv_attention,a.fp16_kv_cache,a.attention_fp16,a.attention_fp16_convert))
        key=next(k for k,v in graph['Layers'].items() if v['op']=='softmax')
        q['quantize_parameters']['@'+key+':out0']={'qtype':'i16','quantizer':'dynamic_fixed_point','rounding':'rtne','fl':15}
    if a.split_kv_attention:
        assert a.compact_gqa
        from split_kv_attention import split_kv_attention
        split_kv_changes.append({'layer':a.first+offset,**split_kv_attention(graph,q,data,
            manifest['heads'],manifest['kv_heads'],manifest['head_dim'],
            a.capacity or manifest['capacity'],a.tokens,side=a.split_kv_side,
            probability_i16=a.split_probability_i16,value_heads=a.split_value_heads,
            value_fp16=a.split_value_fp16,value_transposed=a.split_value_transposed,
            probability_identity=a.split_probability_identity,softmax_i16=a.split_softmax_i16,
            current_matmul=a.split_current_matmul,key_cache_transposed=a.key_cache_transposed)})
    if a.native_rmsnorm:
        assert manifest.get('stable_norm') and manifest.get('norm_version') == 3
        from native_rmsnorm import replace_rmsnorm
        data, change = replace_rmsnorm(graph, q, data)
        norm_changes.append({'layer': a.first + offset, **change})
    if a.key_cache_transposed and not a.split_kv_attention:
        assert a.compact_gqa
        from key_cache_layout import transpose_key_cache
        key_layout_changes.append({'layer':a.first+offset,**transpose_key_cache(graph,q)})
    if a.attention_fp16:
        assert a.compact_gqa and not a.key_cache_transposed
        changes=[]
        for node in graph['Layers'].values():
            if node['op']!='matmul':continue
            for ref in node['inputs']:
                operand=graph['Layers'][ref[1:].rsplit(':',1)[0]]
                assert operand['op']!='input', 'retain external KV precision'
                old=q['quantize_parameters'][ref]
                if old=={'qtype':'float16','quantizer':'float16'}:continue
                changes.append({'tensor':ref,'before':old})
                q['quantize_parameters'][ref]={'qtype':'float16','quantizer':'float16'}
        attention_changes.append({'layer':a.first+offset,'operands':changes})
    if a.attention_fp16_convert:
        assert a.compact_gqa and not a.key_cache_transposed
        changes=[];converters={}
        for key,node in list(graph['Layers'].items()):
            if node['op']!='matmul':continue
            for i,ref in enumerate(node['inputs']):
                old=q['quantize_parameters'][ref]
                if old=={'qtype':'float16','quantizer':'float16'}:continue
                if ref not in converters:
                    name=f'attention_fp16_{len(converters)}'
                    assert name not in graph['Layers']
                    graph['Layers'][name]={'name':name,'op':'dtype_converter','parameters':{},
                        'inputs':[ref],'outputs':['out0']}
                    converters[ref]='@'+name+':out0'
                    q['quantize_parameters'][converters[ref]]={'qtype':'float16','quantizer':'float16'}
                node['inputs'][i]=converters[ref]
                changes.append({'matmul':key,'input':i,'tensor':ref,'before':old,'converted':converters[ref]})
        attention_changes.append({'layer':a.first+offset,'strategy':'explicit-dtype-converter-v1','operands':changes})
    assert not graph.get('quantize_info') and not q.get('customized_quantize_layers')
    original = graph['Layers']
    rename = {key: f'b{offset}_{key}' for key in original}
    hidden = next(k for k, v in original.items() if v['name'] == 'hidden')
    hidden_out = next(k for k, v in original.items() if v['name'] == 'hidden_out')
    hidden_attach = next(k for k, v in original.items() if v['op'] == 'output' and v['inputs'] == ['@' + hidden_out + ':out0'])
    for key in (hidden, hidden_out, hidden_attach):
        assert q['quantize_parameters']['@' + key + ':out0'] == {'qtype': 'float16', 'quantizer': 'float16'}, 'hidden must be FP16'
    removed = ({hidden} if offset else set()) | ({hidden_attach} if offset < a.width - 1 else set())
    def reference(value):
        name, port = value[1:].rsplit(':', 1)
        if offset and name == hidden:
            return '@' + previous_hidden + ':' + port
        return '@' + rename[name] + ':' + port
    for key, value in original.items():
        if key in removed:
            continue
        value = copy.deepcopy(value)
        value['name'] = f'b{offset}_' + value['name']
        value['inputs'] = [reference(v) for v in value.get('inputs', [])]
        nodes[rename[key]] = value
        if value['op'] == 'output':
            output_nodes.append((offset, key == hidden_attach, rename[key]))
    for key, value in q['quantize_parameters'].items():
        name, port = key[1:].rsplit(':', 1)
        if name not in removed:
            quant['@' + rename[name] + ':' + port] = value
    for key, value in data.items():
        if key not in removed:
            weights[rename[key]] = value
    meta = yaml.safe_load((folder / 'tensor-inputmeta.yml').read_text())
    for port in meta['input_meta']['databases'][0]['ports']:
        if offset and port['lid'] == hidden:
            continue
        port = copy.deepcopy(port)
        port['shape']=original[port['lid']]['parameters']['shape']
        port['lid'] = rename[port['lid']]
        ports.append(port)
    sources.append({'layer': a.first + offset, 'folder': str(folder.resolve().relative_to(root)),
        'sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   for name in ('decoder.json', 'decoder.data', 'decoder.quantize')}})
    previous_hidden = rename[hidden_out]
# Explicit output order: final hidden, then K/V for each layer.
outputs = [name for _, is_hidden, name in output_nodes if is_hidden]
outputs += [name for _, is_hidden, name in output_nodes if not is_hidden]
nodes = {**{name: nodes[name] for name in outputs}, **{k: v for k, v in nodes.items() if k not in outputs}}
assert len(ports) == 1 + 5 * a.width and len(outputs) == 1 + 2 * a.width
for value in nodes.values():
    assert all(v[1:].rsplit(':', 1)[0] in nodes for v in value.get('inputs', []))
graph['Layers'] = nodes
graph['MetaData']['Name'] = f'decoder_fusion_{a.width}'
(out / 'decoder.json').write_text(json.dumps(graph, indent=2) + '\n')
with (out / 'decoder.data').open('wb') as stream:
    np.save(stream, weights)
(out / 'decoder.quantize').write_text(yaml.safe_dump({'version': 2, 'quantize_parameters': quant, 'customized_quantize_layers': {}}, sort_keys=False))
meta['input_meta']['databases'][0]['ports'] = ports
(out / 'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta, sort_keys=False))
container = 'opi-fusion-' + uuid.uuid4().hex[:12]
command = ['docker', 'run', '--rm', '--name', container, '--network', 'none', '--cpus', '4', '--memory', '8g',
    '-e', 'TF_CPP_MIN_LOG_LEVEL=3', '-e', 'OMP_NUM_THREADS=4', '-e', 'VSIMULATOR_SHADER_CORE_COUNT=1',
    '-e', 'VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B', '-v', str(root) + ':/work',
    '-w', '/work/' + str(relative), 'ubuntu-npu:v2.0.10.2', 'python3',
    '/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py', 'export', 'ovxlib', '--model', 'decoder.json',
    '--model-data', 'decoder.data', '--model-quantize', 'decoder.quantize', '--dtype', 'quantized',
    '--with-input-meta', 'tensor-inputmeta.yml', '--target-ide-project', 'linux64', '--pack-nbg-unify',
    '--optimize', 'VIP9000NANODI_PLUS_PID0X1000003B', '--viv-sdk', '/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools',
    '--output-path', 'export/decoder']
started = time.monotonic()
if a.node_trace:
    command[2:2] = ['-e', 'VIV_VX_DUMP_NODE_TRACE_DB=1']
for option in a.sdk_option:
    command[2:2] = ['-e', option]
with (out / 'export.log').open('w') as log:
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
        status = result.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log, stderr=subprocess.STDOUT, timeout=15)
        status = 124
report = {'first_layer': a.first, 'fusion_width': a.width, 'sources': sources,
    'fc_input_int8':a.fc_input_int8,'fc_input_range_bits':a.fc_input_range_bits,'fc_input_changes':fc_input_changes,
    'fc_weight_dfp':a.fc_weight_dfp,'fc_weight_dfp_method':a.fc_weight_dfp_method,'fc_weight_changes':fc_weight_changes,
    'mlp_product_headroom':a.mlp_product_headroom,'mlp_product_changes':product_range_changes,
    'node_trace':a.node_trace,
    'fp16_kv_cache':a.fp16_kv_cache,'fp16_kv_changes':fp16_kv_changes,
    'attention_probability_i16':a.attention_probability_i16,
    'split_kv_attention':a.split_kv_attention,'split_kv_changes':split_kv_changes,
    'split_kv_side':a.split_kv_side,
    'split_probability_i16':a.split_probability_i16,
    'split_value_heads':a.split_value_heads,
    'split_value_fp16':a.split_value_fp16,
    'split_value_transposed':a.split_value_transposed,
    'split_probability_identity':a.split_probability_identity,
    'split_softmax_i16':a.split_softmax_i16,
    'split_current_matmul':a.split_current_matmul,
    'sdk_options':a.sdk_option,
    'value_output_fp16':a.value_output_fp16,'value_perchannel':a.value_perchannel,
    'value_layout_fp16':a.value_layout_fp16,
    'value_layout_sliced':a.value_layout_sliced,
    'native_rmsnorm':a.native_rmsnorm,'norm_changes':norm_changes,
    'key_cache_transposed':a.key_cache_transposed,'key_layout_changes':key_layout_changes,
    'attention_fp16':a.attention_fp16,'attention_changes':attention_changes,
    'attention_fp16_convert':a.attention_fp16_convert,
    'capacity':a.capacity or manifest['capacity'],'query_tokens':a.tokens,
    'compact_gqa': a.compact_gqa, 'layout_changes': layout_changes,
    'scope': 'calibrated graph fusion with explicit recorded layout/precision overrides; no recalibration',
    'input_order': [v['lid'] for v in ports], 'output_order': outputs,
    'exit_code': status, 'seconds': time.monotonic() - started,
    'binaries': [{'path': str(f.relative_to(out)), 'bytes': f.stat().st_size} for f in out.rglob('network_binary.nb')]}
if status == 0:
    report.update(verify_contract(out, a.width))
(out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(status)
