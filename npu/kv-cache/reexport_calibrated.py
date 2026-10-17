#!/usr/bin/env python3
"""Recompile a preserved calibrated ACUITY graph without changing its weights."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    names = ('decoder.json', 'decoder.data', 'decoder.quantize', 'tensor-inputmeta.yml')
    for name in names:
        if not (source / name).is_file():
            parser.error('Missing calibrated input: ' + str(source / name))
    record = source / 'conversion.json'
    if record.is_file() and json.loads(record.read_text()).get('sdk_options'):
        parser.error('This runner supports the default SDK options only')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name in names:
        shutil.copy2(source / name, out / name)
        inputs[name] = sha256(out / name)
        if inputs[name] != sha256(source / name):
            raise RuntimeError('Input copy mismatch: ' + name)
    image = 'ubuntu-npu:v2.0.10.2'
    target = 'VIP9000NANODI_PLUS_PID0X1000003B'
    image_info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
    container = 'opi-wsl-reexport-' + uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--name', container, '--network', 'none',
               '--cpus', '4', '--memory', '8g', '-e', 'TF_CPP_MIN_LOG_LEVEL=3',
               '-e', 'OMP_NUM_THREADS=4', '-e', 'VSIMULATOR_SHADER_CORE_COUNT=1',
               '-e', 'VSIMULATOR_CONFIG=' + target, '-v', str(out) + ':/work',
               '-w', '/work', image, 'python3', '/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py',
               'export', 'ovxlib', '--model', 'decoder.json', '--model-data', 'decoder.data',
               '--model-quantize', 'decoder.quantize', '--dtype', 'quantized',
               '--with-input-meta', 'tensor-inputmeta.yml', '--target-ide-project', 'linux64',
               '--pack-nbg-unify', '--optimize', target,
               '--viv-sdk', '/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools',
               '--output-path', 'export/decoder']
    started = time.monotonic()
    with (out / 'export.log').open('w') as log:
        try:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '-t', '2', container], stdout=log,
                           stderr=subprocess.STDOUT, timeout=20, check=False)
            status = 124
    binary = out / 'export_nbg_unify/network_binary.nb'
    prior = source / 'export_nbg_unify/network_binary.nb'
    result = dict(source=str(source), output=str(out), inputs_sha256=inputs,
                  image=image, image_id=image_info['Id'], image_rootfs=image_info['RootFS'],
                  target=target, command=command, exit_code=status,
                  seconds=time.monotonic() - started, board_validated=False)
    if status == 0 and binary.is_file():
        result.update(binary_sha256=sha256(binary), binary_bytes=binary.stat().st_size)
        if prior.is_file():
            result.update(archived_binary_sha256=sha256(prior),
                          binary_matches_archive=sha256(prior) == result['binary_sha256'])
    else:
        status = status or 1
        result['exit_code'] = status
    (out / 'reexport-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
