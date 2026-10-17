#!/usr/bin/env python3
"""Independently verify completed full-model runs against a frozen candidate."""
import argparse
import json
import statistics
from pathlib import Path


def verify(candidate, result, repeats):
    config = json.loads((candidate / 'reproduce.json').read_text())
    expected = json.loads((candidate / 'expected-tokens.json').read_text())
    rows = []
    fields = ('decoder_layers', 'head_on_npu', 'capacity', 'prompt_tokens',
              'generated_tokens', 'fixed_feedback', 'cached_prefix_tokens',
              'new_prompt_tokens', 'prefill_batch_size', 'grouped_head',
              'decoder_group_size', 'fusion_width')
    for name in ['warmup'] + ['measure-' + str(i) for i in range(1, repeats + 1)]:
        row = json.loads((result / name / 'tokens.json').read_text())
        if row['tokens'] != expected['tokens']:
            raise ValueError(name + ': output tokens differ from archive')
        for key in fields:
            if row.get(key) != expected.get(key):
                raise ValueError(name + ': configuration differs: ' + key)
        if expected.get('fixed_feedback') and row.get('feedback_tokens') != expected['feedback_tokens']:
            raise ValueError(name + ': feedback differs from archive')
        if row.get('driver_after_release_bytes') != 0:
            raise ValueError(name + ': NPU memory was not completely released')
        if not row.get('performance_mode'):
            raise ValueError(name + ': expected performance measurement mode')
        rows.append(row)
    measured = rows[1:]
    return dict(candidate=config['id'], verified=True, repeats=repeats,
                scope='Frozen full-depth token trajectory; not independent model quality acceptance',
                decode_median=statistics.median(r['decode_tokens_per_second'] for r in measured),
                ttft_median=statistics.median(r['ttft_seconds'] for r in measured),
                warmup=rows[0], runs=measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    report = verify(args.candidate, args.result, args.repeats)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('warmup', 'runs')}, indent=2))


if __name__ == '__main__':
    main()
