"""Rebuild the 15.598 tok/s stack from fresh Q8 calibration; compare archived hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-block-only', action='store_true')
    a = p.parse_args()
    source, historical, output = a.source.resolve(), a.historical.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    root = source.parent
    m = json.loads((source / 'trained-manifest.json').read_text())
    assert len(m['layers']) == 24 and m['vocab_size'] == 151674
    state = {'status': 'building', 'fresh_source': str(source), 'historical': str(historical), 'comparisons': []}
    def save():
        (output / 'build-state.json').write_text(json.dumps(state, indent=2) + '\n')
    def run(name, command, result):
        if result.exists():
            assert json.loads(result.read_text())['exit_code'] == 0, result
            return
        state['active_stage'] = name
        save()
        print('START', name, flush=True)
        with (output / (name + '.log')).open('x') as log:
            subprocess.run([sys.executable, *map(str, command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    base = source / 'peak155-base'
    base.mkdir(exist_ok=True)
    for i in range(28):
        dest = base / f'l{i}'
        if i < 24 and i not in (2, 3):
            if not dest.exists():
                dest.symlink_to(source / 'w8-balanced' / f'l{i}', target_is_directory=True)
            continue
        # Historical l2/l3 reports have stale range labels. These values reproduce
        # their actual quantize tables and both historical NBG hashes exactly.
        flags = (['--value-dfp', '--float-residual', '--float-sensitive', '--activation-headroom-bits', '0',
                  '--float-mlp', '--int8-mlp-projections', 'gate', 'up', '--mlp-headroom-bits', '1'] if i in (2, 3) else
                 ['--expected-linears', '1', '--weight-mode', 'dfp', '--dfp-method', 'mse', '--activation-headroom-bits', '2'])
        run(f'base-{i}', [scripts / 'compile_weight_int8.py', '--results-root', root,
                         '--base', source / 'int16' / f'l{i}', '--output', dest, *flags], dest / 'conversion.json')
        if a.first_block_only:
            break
    for tokens, variant in [(1, 'w8-fused2-c1280-cap1-headmse'), (16, 'w8-fused2-c1280-t16-cap1-productwide')]:
        target = source / variant
        target.mkdir(exist_ok=True)
        records = []
        for i in range(0, 24, 2):
            dest = target / f'l{i}'
            ranges = [str(v) for layer in (19, 20) if i <= layer < i + 2 for v in ('--mlp-product-headroom', layer, 1)]
            run(f'{tokens}-block-{i}', [scripts / 'compile_fused_decoder.py', '--results-root', root,
                '--source', source, '--variant', base.name, '--first', i, '--width', '2', '--output', dest,
                '--capacity', '1280', '--tokens', tokens, '--compact-gqa', '--split-kv-attention',
                '--split-softmax-i16', '--split-current-matmul', '--fc-weight-dfp', 'gateup',
                '--fc-weight-dfp-method', 'mse', '--fc-weight-dfp-max-extra-bits', '1', *ranges], dest / 'conversion.json')
            old = historical / variant / f'l{i}'
            comparison = {'variant': variant, 'layer': i, 'files': {}}
            for name in ('decoder.json', 'decoder.data', 'decoder.quantize', 'export_nbg_unify/network_binary.nb'):
                x, y = sha(dest / name), sha(old / name)
                comparison['files'][name] = {'fresh_sha256': x, 'historical_sha256': y, 'equal': x == y}
            comparison['quantize_semantically_equal'] = yaml.safe_load((dest / 'decoder.quantize').read_text()) == yaml.safe_load((old / 'decoder.quantize').read_text())
            state['comparisons'].append(comparison)
            save()
            print('COMPARE', variant, i, {k:v['equal'] for k,v in comparison['files'].items()}, flush=True)
            assert comparison['files']['export_nbg_unify/network_binary.nb']['equal'], 'historical NBG mismatch; stop before deployment'
            records.append({'layer': i, 'exit_code': 0, 'fusion_width': 2, 'capacity': 1280,
                            'query_tokens': tokens, 'key_cache_transposed': False,
                            'path': str(dest / 'export_nbg_unify/network_binary.nb')})
            if a.first_block_only:
                return
        for i in range(24, 28):
            dest = target / f'l{i}'
            if not dest.exists():
                dest.symlink_to(base / f'l{i}', target_is_directory=True)
            records.append({'layer': i, 'exit_code': 0, 'fusion_width': 1,
                            'path': str(dest / 'export_nbg_unify/network_binary.nb')})
        (target / 'stack-conversion.json').write_text(json.dumps(records, indent=2) + '\n')
    for i in range(24, 28):
        x = sha(base / f'l{i}' / 'export_nbg_unify/network_binary.nb')
        y = sha(historical / 'w8-fused2-c1280-cap1-headmse' / f'l{i}' / 'export_nbg_unify/network_binary.nb')
        state['comparisons'].append({'head': i, 'fresh_sha256': x, 'historical_sha256': y, 'equal': x == y})
    assert all(x.get('equal', x.get('files', {}).get('export_nbg_unify/network_binary.nb', {}).get('equal')) for x in state['comparisons'])
    state['status'] = 'compiled-awaiting-board-validation'
    save()
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
