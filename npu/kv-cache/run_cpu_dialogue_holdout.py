"""Collect actual board llama.cpp natural answers on the frozen holdout.

Each backend follows its own generated history. Keep this acceptance-only;
do not use the answers for quantizer tuning. Host tokenization is explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import time

from board_host_lock import lock_board_from_host
from resident_http import ChatCodec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'retain previous evidence'
    scripts = Path(__file__).resolve().parent
    raw = (a.reference / 'state.json').read_bytes()
    reference = json.loads(raw)
    suite_path = scripts / 'quality-holdout-v1.json'
    assert reference['status'] == 'complete-reference-only'
    assert hashlib.sha256(suite_path.read_bytes()).hexdigest() == reference['holdout_sha256']
    model = Path(reference['model'])
    codec = ChatCodec(model, reference['system'])
    is_smol = reference['layers'] == 32
    cfg = {'threads': 2, 'affinity': '6,7', 'ubatch': 512, 'flash': 'off' if is_smol else 'auto'}
    a.output.mkdir(parents=True)
    (a.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    report = {'status': 'waiting-for-board', 'scope': 'board llama.cpp Q8 natural quality; not speed search',
              'configuration': cfg, 'holdout_sha256': reference['holdout_sha256'],
              'reference_sha256': hashlib.sha256(raw).hexdigest(), 'model': str(model),
              'layers': reference['layers'], 'system': reference['system'], 'cases': [],
              'human_quality_review': 'pending'}
    def save():
        tmp = a.output / 'state.tmp'
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(a.output / 'state.json')
    save()
    lock_board_from_host()
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
    assert a.output.name.replace('-', '').isalnum()
    remote = '/opt/opi-vlm/npu-kv-probe/' + a.output.name
    subprocess.run(ssh + ['mkdir ' + shlex.quote(remote)], check=True, timeout=30)
    subprocess.run(scp + [str(scripts / 'cpu_dialogue_worker.py'), str(a.output / 'config.json'),
                         'root@192.168.88.29:' + remote + '/'], check=True, timeout=60)
    command = ['python3', '-u', remote + '/cpu_dialogue_worker.py', '--model',
               '/opt/opi-vlm/models/' + model.parent.name + '/' + model.name,
               '--config', remote + '/config.json', '--output', remote + '/result']
    with (a.output / 'transport.log').open('xb') as log:
        process = subprocess.Popen(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock ' + shlex.join(command)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        def event(timeout=400):
            nonlocal buffer
            deadline = time.monotonic() + timeout
            while b'\n' not in buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('CPU quality worker response')
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError('CPU quality worker disconnected')
                buffer.extend(block)
                assert len(buffer) <= 2**20
            line, _, rest = buffer.partition(b'\n')
            buffer = bytearray(rest)
            data = json.loads(line)
            assert data.get('event') != 'error', data
            return data
        def send(data):
            process.stdin.write((json.dumps(data) + '\n').encode())
            process.stdin.flush()
        try:
            ready = event()
            assert ready['event'] == 'ready'
            digest = hashlib.sha256()
            with model.open('rb') as stream:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    digest.update(chunk)
            assert ready['model_sha256'] == digest.hexdigest()
            report.update(status='running', worker=ready)
            save()
            for case in json.loads(suite_path.read_text())['cases']:
                messages = []
                for turn, text in enumerate(case['turns'], 1):
                    messages.append({'role': 'user', 'content': text})
                    prompt = codec.encode(messages)
                    ident = case['id'] + f'-turn{turn}'
                    send({'id': ident, 'tokens': prompt, 'steps': reference['steps_limit'], 'reset': turn == 1})
                    result = event()
                    assert result['event'] == 'done' and result['id'] == ident
                    tokens = result['result']['tokens']
                    answer = codec.decode(tokens)
                    row = {'case': ident, 'messages': list(messages), 'prompt_tokens': prompt,
                           'answer': answer, 'checks': case['checks'], **result}
                    report['cases'].append(row)
                    save()
                    print(ident, len(prompt), len(tokens), flush=True)
                    if not answer:
                        break  # Preserve premature EOS; do not fabricate an assistant turn.
                    messages.append({'role': 'assistant', 'content': answer})
            send({'event': 'close'})
            closed = event(30)
            process.stdin.close()
            assert closed['event'] == 'closed' and not closed['thermal_abort']
            assert process.wait(timeout=30) == 0
            report.update(status='complete-natural-generation', closed=closed)
            save()
        except BaseException as exc:
            report.update(status='failed', error=str(exc))
            save()
            raise
        finally:
            selector.close()
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == '__main__':
    main()
