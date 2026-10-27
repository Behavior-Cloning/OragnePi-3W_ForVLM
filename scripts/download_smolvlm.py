#!/usr/bin/env python3
"""Pinned GGUF downloads; bounded parallel ranges, SHA-256 verification, atomic publish."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

REVISION = 'b9e4379657e1450d04d02eec8e345667265b0a00'
FILES = [
    ('SmolVLM-256M-Instruct-Q8_0.gguf', 175054528,
     '2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65'),
    ('mmproj-SmolVLM-256M-Instruct-f16.gguf', 190031616,
     '0802360aca1748f112ea510b8ff277c65b1361c8ef30ed89b83c9c7a60d08e96'),
]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'models/smolvlm-256m')
    parser.add_argument('--mirror', default='https://hf-mirror.com')
    parser.add_argument('--source-revision', help='Alternate mirror revision; expected size and SHA still come from manifest')
    parser.add_argument('--no-proxy', default='hf-mirror.com', help='curl proxy bypass hosts (use * for direct downloads)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--model', default='smolvlm-256m')
    args = parser.parse_args()
    selected = json.loads(args.manifest.read_text())[args.model] if args.manifest else {
        'repository': 'ggml-org/SmolVLM-256M-Instruct-GGUF', 'revision': REVISION, 'files': FILES}
    if args.model == 'qwen25-15b-q4' or selected.get('status') == 'retired':
        parser.error('This historical model is retired and no longer downloaded')
    args.directory.mkdir(parents=True, exist_ok=True)
    lock = (args.directory / '.download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    chunk_size = 8 * 1024 * 1024
    for name, size, expected in selected['files']:
        if args.only and name != args.only:
            continue
        target = args.directory / name
        if target.exists() and target.stat().st_size == size and digest(target) == expected:
            print(f'Already verified: {name}', flush=True)
            continue
        chunks = args.directory / (name + '.chunks')
        chunks.mkdir(exist_ok=True)
        revision = args.source_revision or selected['revision']
        url = f"{args.mirror}/{selected['repository']}/resolve/{revision}/{name}"

        def fetch(index):
            start = index * chunk_size
            end = min(size, start + chunk_size) - 1
            part = chunks / f'{index:04d}'
            if part.exists() and part.stat().st_size == end - start + 1:
                return part
            temporary = part.with_suffix('.tmp')
            tail = part.with_suffix('.tail')
            headers = part.with_suffix('.headers')
            for attempt in range(10):
                have = temporary.stat().st_size if temporary.exists() else 0
                if have == end - start + 1:
                    temporary.replace(part)
                    print(f'{name}: chunk {index + 1}/{(size + chunk_size - 1) // chunk_size}', flush=True)
                    return part
                if have > end - start + 1:
                    raise RuntimeError(f'Invalid partial chunk size: {temporary}')
                result = subprocess.run([
                    'curl', '--noproxy', args.no_proxy, '-fsSL',
                    '--connect-timeout', '20', '--max-time', '1800',
                    '--speed-time', '90', '--speed-limit', '500',
                    '--range', f'{start + have}-{end}', '-D', str(headers), '-o', str(tail), url,
                ], capture_output=True, text=True)
                expected_range = f'bytes {start + have}-{end}/{size}'
                ranges = re.findall(r'^content-range:\s*(.+)$', headers.read_text(), re.I | re.M) if headers.exists() else []
                if ranges and ranges[-1].strip() == expected_range and tail.exists() and tail.stat().st_size <= end - start + 1 - have:
                    with temporary.open('ab') as output, tail.open('rb') as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(block)
                    if temporary.stat().st_size == end - start + 1:
                        temporary.replace(part)
                        print(f'{name}: chunk {index + 1}/{(size + chunk_size - 1) // chunk_size}', flush=True)
                        return part
                print(f'Retrying {name} chunk {index}: {result.stderr[-160:]}', flush=True)
                time.sleep(min(2 ** attempt, 15))
            raise RuntimeError(f'Download incomplete: {name}, chunk {index}; rerun to resume')

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
            parts = list(pool.map(fetch, range((size + chunk_size - 1) // chunk_size)))
        assembled = target.with_suffix('.gguf.assembling')
        with assembled.open('wb') as output:
            for part in parts:
                with part.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block)
        if digest(assembled) != expected:
            raise RuntimeError(f'SHA-256 mismatch: {assembled}; do not deploy this file')
        os.replace(assembled, target)
        print(f'VERIFIED {expected} {name}', flush=True)


if __name__ == '__main__':
    main()
