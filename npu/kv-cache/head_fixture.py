"""Single output-head shard probe; calibration input, never holdout acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import gguf
import numpy as np
import torch
from exported_int8 import apply_exported_head
from trained_probe import LMHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--hidden', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    graph = a.graph.resolve()
    m = json.loads((a.source / 'trained-manifest.json').read_text())
    shard = next(s for s in m['head_shards'] if s['layer'] == a.layer)
    for name in ('decoder.json', 'decoder.data', 'decoder.quantize', 'conversion.json', 'export_nbg_unify'):
        (a.output / name).symlink_to(graph / name, target_is_directory=(graph / name).is_dir())
    (a.output / 'graphs').mkdir()
    (a.output / 'graphs' / f'l{a.layer}').symlink_to(graph, target_is_directory=True)
    hidden = np.load(a.hidden).astype(np.float32).reshape(1, 1, m['dim'])
    hidden.tofile(a.output / 'probe-input.f32')
    reader = gguf.GGUFReader(a.model)
    arch = reader.fields['general.architecture'].contents()
    eps = reader.fields[arch + '.attention.layer_norm_rms_epsilon'].contents()
    head = LMHead(reader, eps, stable_norm=True, norm_version=3).eval()
    start, end = shard['vocab_start'], shard['vocab_start'] + shard['vocab_count']
    with torch.inference_mode():
        original = head(torch.from_numpy(hidden))[..., start:end].numpy()
        records = apply_exported_head(head, a.output, 'graphs', {**m, 'head_shards': [shard]})
        expected = head(torch.from_numpy(hidden))[..., start:end].numpy()
    expected.tofile(a.output / 'probe-reference.f32')
    original.tofile(a.output / 'probe-original-reference.f32')
    report = {'scope': 'single vocabulary shard, calibration hidden; not independent quality or full-model speed',
              'layer': a.layer, 'vocab_start': start, 'vocab_count': end - start,
              'input_source': str(a.hidden), 'input_sha256': hashlib.sha256(a.hidden.read_bytes()).hexdigest(),
              'outputs': [('logits', end - start)], 'constants': records}
    (a.output / 'probe-fixture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
