"""Join calibrated ACUITY decoder graphs without changing their quantizers.

Only the intermediate hidden input/output boundary is removed. Every layer
keeps its own KV, mask and RoPE ports, so no scale compatibility is assumed.
"""
import argparse
import copy
import hashlib
import json
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
a = p.parse_args()
root = Path(__file__).resolve().parents[2] / 'npu/results'
out = a.output.resolve()
relative = out.relative_to(root)
out.mkdir(parents=True, exist_ok=False)
nodes, weights, quant, ports, output_nodes = {}, {}, {}, [], []
sources = []
layout_changes = []
manifest = json.loads((a.source / 'trained-manifest.json').read_text())
previous_hidden = None
for offset in range(a.width):
    folder = a.source / a.variant / f'l{a.first + offset}'
    graph = json.loads((folder / 'decoder.json').read_text())
    q = yaml.safe_load((folder / 'decoder.quantize').read_text())
    data = np.load(folder / 'decoder.data', allow_pickle=True).item()
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
with (out / 'export.log').open('w') as log:
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
        status = result.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log, stderr=subprocess.STDOUT, timeout=15)
        status = 124
report = {'first_layer': a.first, 'fusion_width': a.width, 'sources': sources,
    'value_output_fp16':a.value_output_fp16,'value_perchannel':a.value_perchannel,
    'value_layout_fp16':a.value_layout_fp16,
    'value_layout_sliced':a.value_layout_sliced,
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
