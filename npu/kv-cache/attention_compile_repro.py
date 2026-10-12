"""Small standalone reproducer for the split-V dynamic GEMM compiler error.

No model weights are needed. Isolate probability slicing from whether V is
an external input or an internally produced concatenation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--prob-source', choices=('direct', 'slice', 'abs'), default='direct')
p.add_argument('--value-source', choices=('direct', 'concat'), default='direct')
p.add_argument('--capacity', type=int, default=1280)
p.add_argument('--kv', type=int, default=5)
p.add_argument('--rows', type=int, default=3)
p.add_argument('--softmax',action='store_true')
p.add_argument('--complete-split-v',action='store_true')
p.add_argument('--prob-int16',action='store_true')
p.add_argument('--current-matmul',action='store_true')
a = p.parse_args()
assert 2 <= a.capacity <= 8192 and 1 <= a.kv <= 16 and 1 <= a.rows <= 32
assert not a.complete_split_v or a.prob_source != 'direct'
root = Path(__file__).resolve().parents[1] / 'results'
out = a.output.resolve(); rel = out.relative_to(root)
out.mkdir(parents=True, exist_ok=False)
nodes = {}; quant = {}; ports = []; feeds = []
fp16 = {'qtype': 'float16', 'quantizer': 'float16'}
probquant = {'qtype': 'i16', 'quantizer': 'dynamic_fixed_point', 'rounding': 'rtne', 'fl': 15} if a.prob_int16 else fp16
i16 = {'qtype': 'i16', 'quantizer': 'dynamic_fixed_point', 'rounding': 'rtne', 'fl': 10}
outquant = {**i16, 'fl': 12}
rng = np.random.default_rng(20261012)
c, kv, rows, hd = a.capacity, a.kv, a.rows, 64
probs = rng.uniform(.01, 1, (1, kv, rows, c + (a.prob_source != 'direct'))).astype(np.float32)
if a.softmax:
    logits = probs * 5
    probs = np.exp(logits - logits.max(-1, keepdims=True))
probs /= probs.sum(-1, keepdims=True)
values = rng.normal(0, .3, (1, kv, c, hd)).astype(np.float32)


def node(name, op, inputs, params, q):
    key = name + '_' + str(len(nodes) + 1)
    nodes[key] = dict(name=name, op=op, parameters=params, inputs=inputs, outputs=['out0'])
    ref = '@' + key + ':out0'; quant[ref] = q.copy()
    return ref


def inp(name, data, q):
    ref = node(name, 'input', [], {'shape': list(data.shape)}, q)
    ports.append(dict(lid=ref[1:].split(':')[0], category='undefined', dtype='float32', sparse=False,
        tensor_name=None, layout='nchw', shape=list(data.shape), fitting='scale', redirect_to_output=False,
        preprocess=dict(reverse_channel=False, scale=1., preproc_node_params=dict(add_preproc_node=False, preproc_type='TENSOR'))))
    feeds.append((name, data))
    return ref


pr = inp('probabilities', logits if a.softmax else probs, fp16 if a.softmax else probquant)
if a.value_source == 'direct':
    vr = inp('values', values, i16)
else:
    v0 = inp('values_past', values[:, :, :-1].copy(), i16)
    v1 = inp('values_new', values[:, :, -1:].copy(), i16)
    vr = node('values_concat', 'concat', [v0, v1], {'dim': 2}, i16)
if a.softmax:
    pr = node('prob_softmax', 'softmax', [pr], {'sf_axis': -1, 'beta': 1.0}, probquant)
all_probs = pr
if a.prob_source != 'direct':
    pr = node('prob_slice', 'slice', [pr], {'begin': [0, 0, 0, 0], 'size': [1, kv, rows, c]}, probquant)
if a.prob_source == 'abs':
    pr = node('prob_abs', 'abs', [pr], {}, probquant)
mm = node('av', 'matmul', [pr, vr], {'transpose_a': False, 'transpose_b': False}, outquant)
new_v = None
if a.complete_split_v:
    new_v = rng.normal(0, .3, (1, kv, 1, hd)).astype(np.float32)
    nv = inp('value_new', new_v, i16)
    pn = node('prob_current', 'slice', [all_probs], {'begin': [0, 0, 0, c], 'size': [1, kv, rows, 1]}, probquant)
    current = node('value_current', 'matmul' if a.current_matmul else 'multiply', [pn, nv],
                   {'transpose_a': False, 'transpose_b': False} if a.current_matmul else {}, outquant)
    mm = node('sum_values', 'add', [mm, current], {}, outquant)
node('result', 'output', [mm], {}, outquant)
graph = dict(MetaData=dict(Name='attention_repro', AcuityVersion='6', Platform='tensorflow', Org_Platform='onnx'), Layers=nodes)
(out / 'decoder.json').write_text(json.dumps(graph, indent=2) + '\n')
(out / 'decoder.quantize').write_text(yaml.safe_dump(dict(version=2, quantize_parameters=quant, customized_quantize_layers={})))
(out / 'tensor-inputmeta.yml').write_text(yaml.safe_dump(dict(input_meta=dict(databases=[dict(path='dataset.txt', type='TEXT', ports=ports)]))))
with (out / 'decoder.data').open('wb') as f:np.save(f, {})
np.concatenate([d.ravel() for _, d in feeds]).tofile(out / 'probe-input.f32')
expected = probs[..., :c] @ values
if new_v is not None:expected += probs[..., c:] * new_v
expected.astype(np.float32).tofile(out / 'probe-reference.f32')
(out / 'probe-fixture.json').write_text(json.dumps(dict(scope='synthetic compiler reproducer only',
    inputs=[(name, d.size) for name, d in feeds], outputs=[('result', expected.size)]), indent=2) + '\n')
container = 'opi-av-repro-' + uuid.uuid4().hex[:12]
command = ['docker', 'run', '--rm', '--name', container, '--network', 'none', '--cpus', '4', '--memory', '8g',
    '-e', 'TF_CPP_MIN_LOG_LEVEL=3', '-e', 'OMP_NUM_THREADS=4', '-e', 'VSIMULATOR_SHADER_CORE_COUNT=1',
    '-e', 'VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B', '-e', 'VIV_VX_DEBUG_LEVEL=1',
    '-e', 'VIV_VX_DUMP_NODE_TRACE_DB=1', '-v', str(root) + ':/work', '-w', '/work/' + str(rel),
    'ubuntu-npu:v2.0.10.2', 'python3', '/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py',
    'export', 'ovxlib', '--model', 'decoder.json', '--model-data', 'decoder.data', '--model-quantize', 'decoder.quantize',
    '--dtype', 'quantized', '--with-input-meta', 'tensor-inputmeta.yml', '--target-ide-project', 'linux64',
    '--pack-nbg-unify', '--optimize', 'VIP9000NANODI_PLUS_PID0X1000003B',
    '--viv-sdk', '/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools', '--output-path', 'export/decoder']
start = time.monotonic()
with (out / 'export.log').open('w') as log:
    try:status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log, stderr=subprocess.STDOUT, timeout=15)
        status = 124
report = dict(prob_source=a.prob_source, value_source=a.value_source, softmax=a.softmax, complete_split_v=a.complete_split_v, capacity=c, kv=kv, rows=rows,
              prob_int16=a.prob_int16, current_matmul=a.current_matmul, exit_code=status, seconds=time.monotonic()-start)
(out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(status)
