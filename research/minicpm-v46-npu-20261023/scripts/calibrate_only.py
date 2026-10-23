#!/usr/bin/env python3
"""Bounded ACUITY conversion, with retained logs and no board modifications."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'npu/results'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--results-root', type=Path, default=RESULTS, help='common bind-mount root for source and output')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--dtype', choices=['int8', 'int16', 'fp16', 'bf16'], default='int16')
    p.add_argument('--timeout', type=int, default=600)
    args = p.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    results = args.results_root.resolve()
    source_rel = source.relative_to(results)
    out_rel = out.relative_to(results)
    out.mkdir(parents=True, exist_ok=True)
    image = 'ubuntu-npu:v2.0.10.2'
    base = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '4', '--memory', '8g',
            '-e', 'TF_CPP_MIN_LOG_LEVEL=3', '-e', 'OMP_NUM_THREADS=4',
            '-e', 'VSIMULATOR_SHADER_CORE_COUNT=1',
            '-e', 'VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B',
            '-v', f'{results}:/work', '-w', f'/work/{out_rel}', image,
            'python3', '/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py']
    report = {'image': image, 'target': 'VIP9000NANODI_PLUS_PID0X1000003B',
              'dtype': args.dtype, 'source': str(source_rel), 'layer': args.layer, 'stages': []}

    def run(label, argv):
        start = time.monotonic()
        container = 'opi-kv-' + uuid.uuid4().hex[:16]
        with (out / f'{label}.log').open('w') as log:
            try:
                proc = subprocess.run(base[:3] + ['--name', container] + base[3:] + argv,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log,
                               stderr=subprocess.STDOUT, timeout=15)
                report['status'] = 'TIMEOUT'
                report['stages'].append({'stage': label, 'exit': 124, 'seconds': time.monotonic()-start})
                (out / 'conversion.json').write_text(json.dumps(report, indent=2))
                raise SystemExit(f'{label} timed out; container stopped')
        report['stages'].append({'stage': label, 'exit': proc.returncode, 'seconds': time.monotonic() - start})
        (out / 'conversion.json').write_text(json.dumps(report, indent=2))
        print(label, proc.returncode, flush=True)
        if proc.returncode:
            raise SystemExit(f'{label} failed; see {out / (label + ".log")}')

    run('import', ['import', 'onnx', '--model', f'/work/{source_rel}/layer{args.layer}.onnx',
                   '--output-model', 'decoder.json', '--output-data', 'decoder.data'])
    run('inputmeta', ['generate', 'inputmeta', '--model', 'decoder.json', '--input-meta-output', 'inputmeta.yml'])
    meta = yaml.safe_load((out / 'inputmeta.yml').read_text())
    db = meta['input_meta']['databases'][0]
    db['path'] = 'dataset.txt'
    for port in db['ports']:
        port['category'] = 'undefined'
        port['preprocess'] = {'reverse_channel': False, 'scale': 1.,
                              'preproc_node_params': {'add_preproc_node': False, 'preproc_type': 'TENSOR'}}
    (out / 'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta, sort_keys=False))
    dataset = (source / f'layer{args.layer}-dataset.txt').read_text().splitlines()
    (out / 'dataset.txt').write_text('\n'.join(' '.join(f'/work/{source_rel}/{v}' for v in line.split()) for line in dataset) + '\n')
    quantizer, qtype = {'int8': ('dynamic_fixed_point', 'int8'), 'int16': ('dynamic_fixed_point', 'int16'), 'fp16': ('float16', 'float16'),
                        'bf16': ('bfloat16', 'bfloat16')}[args.dtype]
    run('quantize', ['quantize', '--model', 'decoder.json', '--model-data', 'decoder.data',
                     '--model-quantize', 'decoder.quantize', '--device', 'CPU', '--iterations', str(len(dataset)),
                     '--with-input-meta', 'tensor-inputmeta.yml', '--quantizer', quantizer, '--qtype', qtype, '--rebuild'])
    report['status']='CALIBRATED_ONLY'
    (out/'conversion.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
