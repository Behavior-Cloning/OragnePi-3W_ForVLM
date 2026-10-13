"""Board-only JSON worker for natural Q8 llama.cpp quality comparisons.

Run under the common board flock. This does not replace the production server
and does not claim controlled performance measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    cfg = json.loads(a.config.read_text())
    binary = Path('/opt/opi-vlm/bin/llama-server')
    port = 18084
    command = ['taskset', '-c', cfg['affinity'], str(binary), '--model', str(a.model),
               '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '1280',
               '--threads', str(cfg['threads']), '--threads-batch', str(cfg['threads']),
               '--parallel', '1', '--n-gpu-layers', '0', '--batch-size', '2048',
               '--ubatch-size', str(cfg['ubatch']), '--flash-attn', cfg['flash'],
               '--no-context-shift', '--no-warmup']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(path, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=180) as response:
            return json.load(response)
    def emit(data):
        print(json.dumps(data), flush=True)
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                value.update(chunk)
        return value.hexdigest()
    def temperature():
        return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
    import socket
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0, 'isolated quality port already in use'
    def interrupted(signum, frame):
        raise SystemExit('terminated')
    signal.signal(signal.SIGTERM, interrupted)
    aborts = []
    stop = threading.Event()
    with (a.output / 'server.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        def guard():
            while not stop.wait(.5):
                value = temperature()
                if value >= 80000:
                    aborts.append({'temperature_mC': value})
                    process.terminate()
                    return
        thread = threading.Thread(target=guard, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert process.poll() is None, 'isolated llama.cpp exited'
                try:
                    if api('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(.2)
            else:
                raise TimeoutError('CPU quality server readiness')
            emit({'event': 'ready', 'command': command, 'binary_sha256': digest(binary),
                  'model_sha256': digest(a.model), 'configuration': cfg})
            for line in sys.stdin:
                request = json.loads(line)
                if request.get('event') == 'close':
                    break
                tokens, steps = request['tokens'], request['steps']
                assert tokens and 1 <= steps <= 128 and len(tokens) + steps - 1 <= 1280
                assert all(type(t) is int and t >= 0 for t in tokens)
                deadline = time.monotonic() + 300
                while temperature() > 65000:
                    assert not aborts and process.poll() is None
                    assert time.monotonic() < deadline, 'quality cooldown timeout'
                    time.sleep(.5)
                start = time.monotonic()
                before = temperature()
                result = api('/completion', {'prompt': tokens, 'n_predict': steps,
                    'cache_prompt': not request['reset'], 'stream': False,
                    'temperature': 0, 'seed': 42, 'ignore_eos': False,
                    'repeat_penalty': 1.0, 'return_tokens': True})
                assert not aborts and result['tokens'], 'empty CPU token sequence or thermal abort'
                emit({'event': 'done', 'id': request['id'], 'result': result,
                      'elapsed_seconds': time.monotonic() - start,
                      'temperature_start_mC': before, 'temperature_end_mC': temperature()})
            emit({'event': 'closed', 'thermal_abort': aborts})
        except BaseException as exc:
            emit({'event': 'error', 'error': str(exc), 'thermal_abort': aborts})
            raise
        finally:
            stop.set()
            thread.join(timeout=2)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == '__main__':
    main()
