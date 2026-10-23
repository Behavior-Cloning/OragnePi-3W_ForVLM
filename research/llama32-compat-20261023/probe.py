#!/usr/bin/env python3
"""Fixed-shape incremental decoder probe; host validation is NOT NPU success.

Random weights exercise RMSNorm, RoPE, GQA and SwiGLU, not language quality.
Each graph consumes one new hidden vector and persistent KV and emits only
the new KV slice. Unused cache slots are masked. No old token is re-projected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

DIM, HEADS, KV_HEADS, HEAD_DIM = 64, 4, 2, 16
INPUTS = ['hidden', 'past_k', 'past_v', 'mask', 'rope_cos', 'rope_sin']
OUTPUTS = ['hidden_out', 'new_k', 'new_v']


def rope_tables(positions):
    freq = positions.float()[:, None] / (10000 ** (torch.arange(8).float() / 8))
    phase = torch.cat((freq, freq), -1)[None, None]
    return phase.cos(), phase.sin()


def rotate(x, cos, sin):
    return x * cos + torch.cat((-x[..., 8:], x[..., :8]), -1) * sin


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(DIM, DIM, bias=False)
        self.gate = nn.Linear(DIM, 192, bias=False)
        self.up = nn.Linear(DIM, 192, bias=False)
        self.down = nn.Linear(192, DIM, bias=False)
        self.norm1 = nn.Parameter(torch.ones(DIM))
        self.norm2 = nn.Parameter(torch.ones(DIM))

    @staticmethod
    def norm(x, scale):
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * scale

    def project(self, hidden, cos, sin):
        x = self.norm(hidden, self.norm1)
        q = self.q(x).reshape(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).reshape(1, -1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).reshape(1, -1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        return rotate(q, cos, sin), rotate(k, cos, sin), v

    def finish(self, hidden, q, k, v, mask):
        # Contiguous repeats preserve GQA head mapping: k0,k0,k1,k1.
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = q @ k.transpose(-1, -2) / HEAD_DIM ** .5 + mask
        attn = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(1, -1, DIM)
        x = hidden + self.o(attn)
        y = self.norm(x, self.norm2)
        gate = self.gate(y)
        return x + self.down((gate * torch.sigmoid(gate)) * self.up(y))

    def forward(self, hidden, past_k, past_v, mask, rope_cos, rope_sin):
        q, k, v = self.project(hidden, rope_cos, rope_sin)
        y = self.finish(hidden, q, torch.cat((past_k, k), 2),
                        torch.cat((past_v, v), 2), mask)
        return y, k, v

    def full(self, hidden):
        length = hidden.shape[1]
        cos, sin = rope_tables(torch.arange(length))
        q, k, v = self.project(hidden, cos, sin)
        mask = torch.triu(torch.full((length, length), -10000.), diagonal=1)[None, None]
        return self.finish(hidden, q, k, v, mask)


class Cache:
    def __init__(self, capacity, layers=2):
        self.capacity = capacity
        self.k = np.zeros((layers, 1, KV_HEADS, capacity, HEAD_DIM), np.float32)
        self.v = np.zeros_like(self.k)
        self.position = 0

    def reset(self):
        self.k.fill(0)
        self.v.fill(0)
        self.position = 0

    def inputs(self, layer, hidden):
        if self.position >= self.capacity:
            raise ValueError('KV capacity exhausted: explicit reset/re-prefill required')
        mask = np.full((1, 1, 1, self.capacity + 1), -10000., np.float32)
        mask[..., :self.position] = 0
        mask[..., -1] = 0  # Current token is the last graph attention slot.
        cos, sin = rope_tables(torch.tensor([self.position]))
        return dict(zip(INPUTS, [hidden, self.k[layer], self.v[layer], mask, cos.numpy(), sin.numpy()]))

    def update(self, layer, k, v):
        if self.position >= self.capacity:
            raise ValueError('KV capacity exhausted')
        self.k[layer, :, :, self.position:self.position + 1] = k
        self.v[layer, :, :, self.position:self.position + 1] = v


def export_layer(layer, capacity, path):
    dummy = Cache(capacity).inputs(0, np.ones((1, 1, DIM), np.float32))
    torch.onnx.export(layer, tuple(torch.from_numpy(v) for v in dummy.values()), str(path),
                      input_names=INPUTS, output_names=OUTPUTS, opset_version=13,
                      dynamo=False, do_constant_folding=True)
    model = onnx.load(path)
    # Opset 13 graphs use no IR>=10 features; retain broad runtime compatibility.
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capacities', nargs='+', type=int, default=[32, 128])
    args = p.parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
        from onnx.reference import ReferenceEvaluator
    torch.set_num_threads(1)
    torch.manual_seed(20261010)
    layers = [Decoder().eval(), Decoder().eval()]
    report = {'status': 'HOST_VALIDATION_ONLY', 'npu_tested': False,
              'weights': 'deterministic random two-layer decoder; not a trained LLM',
              'features': ['incremental KV', 'RoPE absolute positions', 'GQA', 'RMSNorm', 'SwiGLU',
                           'fixed cache buffers', 'delta-only KV outputs'], 'cases': []}
    args.output.mkdir(parents=True, exist_ok=True)
    report['onnx_backend'] = 'onnxruntime CPU' if ort else 'onnx ReferenceEvaluator CPU'
    if ort:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
    with torch.inference_mode():
        for capacity in args.capacities:
            if capacity < 2:
                raise ValueError('capacity must be >=2')
            folder = args.output / f'c{capacity}'
            folder.mkdir(exist_ok=True)
            sessions = []
            for index, layer in enumerate(layers):
                path = folder / f'layer{index}.onnx'
                export_layer(layer, capacity, path)
                sessions.append(ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
                                if ort else ReferenceEvaluator(str(path)))
            cache = Cache(capacity)
            # Poison all unused slots to detect an accidentally open attention mask.
            cache.k.fill(7)
            cache.v.fill(-11)
            hidden = torch.randn(1, capacity, DIM)
            oracle = hidden
            for layer in layers:
                oracle = layer.full(oracle)
            max_error = 0.
            datasets = [[], []]
            for pos in range(capacity):
                x = hidden[:, pos:pos + 1].numpy()
                for index, session in enumerate(sessions):
                    feed = cache.inputs(index, x)
                    if pos in {0, 1, capacity // 2, capacity - 1}:
                        names = []
                        for name, value in feed.items():
                            filename = f'l{index}_p{pos}_{name}.npy'
                            np.save(folder / filename, value)
                            names.append(filename)
                        datasets[index].append(' '.join(names))
                    x, k, v = session.run(OUTPUTS, feed)
                    cache.update(index, k, v)
                error = float(np.max(np.abs(x - oracle[:, pos:pos + 1].numpy())))
                max_error = max(max_error, error)
                np.testing.assert_allclose(x, oracle[:, pos:pos + 1].numpy(), atol=2e-5, rtol=2e-5)
                cache.position += 1
            try:
                cache.inputs(0, x)
                raise AssertionError('overflow was not rejected')
            except ValueError:
                pass
            cache.reset()
            x = hidden[:, :1].numpy()
            for index, session in enumerate(sessions):
                x, k, v = session.run(OUTPUTS, cache.inputs(index, x))
                cache.update(index, k, v)
            np.testing.assert_allclose(x, oracle[:, :1].numpy(), atol=2e-5, rtol=2e-5)
            for index, entries in enumerate(datasets):
                (folder / f'layer{index}-dataset.txt').write_text('\n'.join(entries) + '\n')
            report['cases'].append({'capacity': capacity, 'tokens_checked': capacity,
                                    'max_abs_error': max_error, 'poison_mask_check': 'PASS',
                                    'overflow_check': 'PASS', 'reset_check': 'PASS',
                                    'cache_bytes_two_layers_fp32': int(cache.k.nbytes + cache.v.nbytes),
                                    'host_delta_write_bytes_per_token': 2 * 2 * KV_HEADS * HEAD_DIM * 4})
    model = onnx.load(folder / 'layer0.onnx')
    def describe(value):
        tensor = value.type.tensor_type
        return {'name': value.name, 'shape': [d.dim_value for d in tensor.shape.dim], 'onnx_dtype': tensor.elem_type}
    manifest = {'inputs': [describe(a) for a in model.graph.input],
                'outputs': [describe(a) for a in model.graph.output],
                'note': 'Shapes shown for final capacity. Quantized NBG metadata must be queried separately.'}
    (args.output / 'tensor-contract.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'host-validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
