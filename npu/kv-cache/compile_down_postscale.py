"""Screen factoring an existing per-channel W8 down projection into DFP + scale.

The original integer coefficients and channel scales are read from exported
compiler constants. No new weight calibration or holdout data is used. This
is a local two-layer experiment, never a complete-model performance claim.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch
import yaml

from exported_graph_data import restore
from fused_contract import verify_contract
from graph_reference import evaluate


def coefficients(folder, name, shape):
    code = (folder/'export/vnn_decoder.c').read_text()
    block = code.split('/* @'+name+':weight */', 1)[1].split('/*', 1)[0]
    match = re.search(r'VSI_NN_TYPE_INT8,\s*(\d+),\s*(\d+)\)', block)
    assert match and 'VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC' in block
    offset, size = map(int, match.groups())
    raw = (folder/'export/decoder.export.data').read_bytes()
    integer = np.frombuffer(raw, np.int8, count=size, offset=offset).reshape(shape[::-1]).T.copy()
    scale_id = re.search(r'attr.dtype.scales = scales_(\d+)', block)[1]
    body = re.search(r'float\s+scales_'+scale_id+r'\[\]\s*=\s*\{([^}]+)\}', code)[1]
    scales = np.asarray([float(v.strip()) for v in body.split(',') if v.strip()], np.float32)
    assert scales.shape == (shape[1],) and np.all(scales > 0)
    return integer, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--intermediate', choices=('i16', 'float16'), default='i16')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]/'results'
    source, out = args.folder.resolve(), args.output.resolve()
    relative = out.relative_to(root)
    out.mkdir(parents=True, exist_ok=False)
    graph = json.loads((source/'decoder.json').read_text())
    data = np.load(source/'decoder.data', allow_pickle=True).item()
    quant = yaml.safe_load((source/'decoder.quantize').read_text())
    old = json.loads((source/'conversion.json').read_text())
    assert old['fusion_width'] == 2 and old['query_tokens'] == 1
    baseline = restore(graph, data, source, old, args.model)
    reference_data = copy.deepcopy(baseline)
    nodes, qp = graph['Layers'], quant['quantize_parameters']
    changes = []
    expected_integers = {}
    for prefix in ('b0_', 'b1_'):
        # Down is the unique expansion-to-hidden FC. K/V are hidden-to-smaller.
        matrices = {k: v['weight'] for k, v in data.items()
                    if k.startswith(prefix) and nodes[k]['op'] == 'fullconnect'}
        name = max(matrices, key=lambda k: matrices[k].shape[0])
        matrix = matrices[name]
        assert matrix.shape[0] > matrix.shape[1]
        assert not np.any(data[name]['bias']) and not np.any(baseline[name]['bias'])
        integer, scales = coefficients(source, name, matrix.shape)
        gain = scales*128
        packed = integer.astype(np.float32)/128
        np.testing.assert_array_equal(packed*gain, baseline[name]['weight'])
        expected_integers[name] = integer
        previous = copy.deepcopy(qp['@'+name+':out0'])
        assert previous['qtype'] == 'i16' and previous['quantizer'] == 'dynamic_fixed_point'
        # ACUITY requires the final underscore-delimited node ID to be numeric.
        scale_name, product_name = prefix+'postscale_12000', prefix+'restored_12001'
        assert scale_name not in nodes and product_name not in nodes
        ref, restored_ref = '@'+name+':out0', '@'+product_name+':out0'
        for node in nodes.values():
            node['inputs'] = [restored_ref if v == ref else v for v in node.get('inputs', [])]
        nodes[scale_name] = {'name': scale_name, 'op': 'variable',
                            'parameters': {'shape': [1, len(gain)], 'is_scalar': False, 'type': 'float32'},
                            'inputs': [], 'outputs': ['out0']}
        nodes[product_name] = {'name': product_name, 'op': 'multiply', 'parameters': {},
                              'inputs': [ref, '@'+scale_name+':out0'], 'outputs': ['out0']}
        data[name]['weight'] = packed
        data[scale_name] = {'data': gain.reshape(1, -1)}
        reference_data[name]['weight'] = packed
        reference_data[scale_name] = {'data': gain.reshape(1, -1)}
        qp['@'+name+':weight'] = {'qtype': 'i8', 'quantizer': 'dynamic_fixed_point',
                                 'fl': 7, 'rounding': 'rtne', 'min_value': float(packed.min()),
                                 'max_value': float(packed.max())}
        qp[restored_ref] = previous
        # The gain constant is explicitly FP16, not silently treated as exact FP32.
        qp['@'+scale_name+':out0'] = {'qtype': 'float16', 'quantizer': 'float16'}
        gain_error = float(np.linalg.norm(gain.astype(np.float16).astype(np.float32)-gain)/np.linalg.norm(gain))
        if args.intermediate == 'float16':
            qp[ref] = {'qtype': 'float16', 'quantizer': 'float16'}
        else:
            # Cover every old channel's representable interval after removing gain.
            extra = math.ceil(math.log2(1/float(gain.min())))
            qp[ref] = {**previous, 'fl': previous['fl']-extra,
                       'max_value': previous['max_value']/float(gain.min()),
                       'min_value': previous['min_value']/float(gain.min())}
        changes.append({'linear': name, 'weight_shape': list(matrix.shape),
                        'integer_sha256': hashlib.sha256(integer.tobytes()).hexdigest(),
                        'gain_min': float(gain.min()), 'gain_max': float(gain.max()),
                        'gain_fp16_relative_l2': gain_error, 'old_output_quant': previous,
                        'intermediate_quant': qp[ref]})
    feeds, offset = {}, 0
    raw = np.fromfile(args.fixture/'probe-input.f32', np.float32)
    for key in old['input_order']:
        node = nodes[key];shape = node['parameters']['shape'];count = int(np.prod(shape))
        feeds[node['name']] = torch.from_numpy(raw[offset:offset+count].reshape(shape).copy())
        offset += count
    assert offset == raw.size
    original_graph = json.loads((source/'decoder.json').read_text())
    with torch.inference_mode():
        expected = evaluate(original_graph, baseline, feeds)
        factored = evaluate(graph, reference_data, feeds)
        for name, value in expected.items():
            torch.testing.assert_close(factored[name], value, rtol=5e-5, atol=2e-5)
        output_names = [nodes[key]['name'] for key in old['output_order']]
        reference = np.concatenate([expected[k].numpy().ravel() for k in output_names])
        np.testing.assert_allclose(reference, np.fromfile(args.fixture/'probe-reference.f32', np.float32),
                                   rtol=5e-4, atol=1e-4)
    reference.tofile(out/'probe-reference.f32')
    shutil.copyfile(args.fixture/'probe-input.f32', out/'probe-input.f32')
    fixture = json.loads((args.fixture/'probe-fixture.json').read_text())
    fixture.update(scope='candidate compared against unchanged baseline exported-weight FP32; all outputs; local probe only',
                   baseline_graph=str(source), reference_is_candidate_exported_oracle=False,
                   exact_integer_weights_preserved=True, postscale_changes=changes)
    (out/'probe-fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    (out/'decoder.json').write_text(json.dumps(graph, indent=2)+'\n')
    with (out/'decoder.data').open('wb') as stream:
        np.save(stream, data)
    (out/'decoder.quantize').write_text(yaml.safe_dump(quant, sort_keys=False))
    shutil.copyfile(source/'tensor-inputmeta.yml', out/'tensor-inputmeta.yml')
    container = 'opi-postscale-'+uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--name', container, '--network', 'none', '--cpus', '4', '--memory', '8g',
               '-e', 'TF_CPP_MIN_LOG_LEVEL=3', '-e', 'OMP_NUM_THREADS=4', '-e', 'VSIMULATOR_SHADER_CORE_COUNT=1',
               '-e', 'VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B', '-v', str(root)+':/work',
               '-w', '/work/'+str(relative), 'ubuntu-npu:v2.0.10.2', 'python3',
               '/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py', 'export', 'ovxlib', '--model', 'decoder.json',
               '--model-data', 'decoder.data', '--model-quantize', 'decoder.quantize', '--dtype', 'quantized',
               '--with-input-meta', 'tensor-inputmeta.yml', '--target-ide-project', 'linux64', '--pack-nbg-unify',
               '--optimize', 'VIP9000NANODI_PLUS_PID0X1000003B', '--viv-sdk', '/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools',
               '--output-path', 'export/decoder']
    start = time.monotonic()
    with (out/'export.log').open('x') as log:
        try:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log, stderr=subprocess.STDOUT, timeout=15)
            status = 124
    record = {**old, 'baseline_graph': str(source), 'postscale_changes': changes,
              'postscale_intermediate': args.intermediate, 'exit_code': status,
              'seconds': time.monotonic()-start, 'scope': 'two-layer scale-factorization probe only',
              'binaries': [{'path': str(f.relative_to(out)), 'bytes': f.stat().st_size,
                            'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                           for f in out.rglob('network_binary.nb')],
              'baseline_conversion_sha256': hashlib.sha256((source/'conversion.json').read_bytes()).hexdigest()}
    if status == 0:
        record.update(verify_contract(out, 2))
        code = (out/'export/vnn_decoder.c').read_text()
        encoded = (out/'export/decoder.export.data').read_bytes()
        for name, integer in expected_integers.items():
            block = code.split('/* @'+name+':weight */', 1)[1].split('/*', 1)[0]
            match = re.search(r'VSI_NN_TYPE_INT8,\s*(\d+),\s*(\d+)\)', block)
            assert match and re.search(r'attr.dtype.fl = 7;', block)
            offset, size = map(int, match.groups())
            actual = np.frombuffer(encoded, np.int8, count=size, offset=offset).reshape(integer.shape[::-1]).T
            np.testing.assert_array_equal(actual, integer)
        record['exported_integer_bytes_verified'] = True
    (out/'conversion.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'exit_code': status, 'changes': changes}), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
