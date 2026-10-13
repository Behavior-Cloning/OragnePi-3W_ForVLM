"""Screen 16/32 query blocks before spending board time on complete stacks."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results'


def main():
    reports = {}
    for model in ('smol500', 'intern1b'):
        folder = ROOT / (model + '-prefill16-20261013')
        for size in (32, 16):
            result = folder / f'probe-t{size}'
            record = result / 'board-validation.json'
            if not record.exists():
                assert not result.exists(), 'retain partial attempt and inspect before retry'
                command = [sys.executable, str(HERE / 'run_tensor_probe.py'),
                           '--folder', str(folder / f'fixture-t{size}'),
                           '--tag', f'{model}-prefill-t{size}-screen-20261013',
                           '--result-dir', str(result), '--repeats', '50', '--warmup', '3',
                           '--cpu-affinity', '6,7']
                print(model, size, 'starting bounded probe', flush=True)
                with (folder / f'probe-t{size}.log').open('x') as log:
                    child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if child.returncode not in (0, 3):
                    child.check_returncode()
            report = json.loads(record.read_text())
            reports[f'{model}-t{size}'] = report
            print(model, size, report['gate'], report['timing']['mean_api_seconds'], flush=True)
    output = ROOT / 'prefill16-screen-20261013.json'
    output.write_text(json.dumps({'scope': 'local block screening; not full-model TTFT', 'runs': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
