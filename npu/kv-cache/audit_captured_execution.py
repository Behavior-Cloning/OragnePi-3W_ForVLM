"""Measure local device error with the device's actual input, not a prefix oracle.

Both references begin with exactly the captured hidden/KV. This separates a
block's own execution error from errors already accumulated before that block.
It is diagnostic only, not a complete-model quality or performance result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    norm = float(np.linalg.norm(b))
    error = float(np.linalg.norm(a-b))
    product = float(np.linalg.norm(a))*norm
    return {'relative_l2': error/norm if norm else None,
            'absolute_l2': error, 'reference_l2': norm,
            'cosine': float(a@b)/product if product else None,
            'max_absolute_error': float(np.max(np.abs(a-b)))}


def audit(fixture, actual_path):
    description = json.loads((fixture/'probe-fixture.json').read_text())
    inputs = fixture/'probe-input.f32'
    # Legacy fixtures preceded optional layout conversion and replay hashes.
    expected_input_hash = description.get('replayed_input_sha256', description['capture_sha256'])
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == expected_input_hash
    captured = Path(description['capture_path'])
    assert hashlib.sha256(captured.read_bytes()).hexdigest() == description['capture_sha256']
    assert actual_path.resolve() == captured.with_name('decode-capture-output.f32').resolve(), 'mismatched capture output'
    paths = {'actual': actual_path, 'exported': fixture/'probe-reference.f32',
             'original': fixture/'probe-original-reference.f32'}
    arrays = {k: np.fromfile(p, np.float32) for k, p in paths.items()}
    count = sum(n for _, n in description['outputs'])
    assert all(a.size == count for a in arrays.values()), 'incomplete capture/reference'
    rows, offset = [], 0
    for name, size in description['outputs']:
        a, e, o = (arrays[k][offset:offset+size] for k in ('actual', 'exported', 'original'))
        rows.append({'name': name, 'count': size,
                     'execution_vs_exported': compare(a, e),
                     'device_vs_original': compare(a, o),
                     'exported_constants_vs_original': compare(e, o)})
        offset += size
    return {'scope': __doc__, 'fixture': str(fixture.resolve()),
            'first_layer': description['first_layer'], 'outputs': rows,
            'sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
            'capture_input_sha256': description['capture_sha256']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'preserve earlier diagnostic'
    result = audit(a.fixture, a.actual)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['outputs'], indent=2))


if __name__ == '__main__':
    main()
