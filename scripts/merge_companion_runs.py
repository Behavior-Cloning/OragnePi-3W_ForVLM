#!/usr/bin/env python3
"""Merge disjoint scenario runs with the same suite/settings, preserving run provenance."""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output')
    runs = [json.loads(p.read_text()) for p in args.inputs]
    reference = runs[0]['plan']
    assert all(r['status'].startswith('COMPLETE') for r in runs), 'Runs still incomplete'
    assert all(r['plan']['suite_sha256'] == reference['suite_sha256'] and r['plan']['settings'] == reference['settings'] for r in runs)
    assert all({m['id'] for m in r['models']} == {m['id'] for m in runs[0]['models']} for r in runs)
    result = copy.deepcopy(runs[0])
    result['source_runs'] = [str(p) for p in args.inputs]
    result['status'] = 'COMPLETE_WITH_ERRORS' if any(r['status'] != 'COMPLETE' for r in runs) else 'COMPLETE'
    result['production_restored'] = all(r.get('production_restored') for r in runs)
    result['plan']['tasks'] = [t for r in runs for t in r['plan']['tasks']]
    ids = [t['id'] for t in result['plan']['tasks']]
    assert len(ids) == len(set(ids)), 'Repeated scenarios must not silently count twice'
    result['plan']['split'] = 'merged'
    result['plan']['excluded_ids'] = []
    result['plan']['turn_count'] = sum(len(t['user_turns']) for t in result['plan']['tasks'])
    for model in result['models']:
        sources = [next(m for m in r['models'] if m['id'] == model['id']) for r in runs]
        assert all(m['manifest'] == model['manifest'] for m in sources)
        model['startups'] = [m.get('startup') for m in sources]
        model['conversations'] = [dict(c, source_run=str(p)) for m, p in zip(sources, args.inputs) for c in m['conversations']]
        model['errors'] = [m['error'] for m in sources if m.get('error')]
    result['notes'].append('Merged disjoint scenario batches; startup column shows first batch only. See startups for both.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
