"""Board-backed HTTP development checks, separate from independent holdout.

Host tokenizer/HTTP and SSH transport are explicit in the result. This is not
an on-board production website or a replacement for controlled speed trials.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

from resident_http import ChatCodec, ChatEngine, make_server
from resident_worker import ResidentWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    reference = args.reference.resolve()
    manifest = json.loads((reference.parents[1] / 'trained-manifest.json').read_text())
    runtime = json.loads((reference / 'tokens.json').read_text())
    assert runtime['decoder_layers'] == manifest['language_layers_total']
    assert json.loads((reference / 'full-validation-w8.json').read_text())['execution_gate_v1']['passed']
    assert args.tag.replace('-', '').isalnum() and args.output.name.replace('-', '').isalnum()
    assert not args.output.exists(), 'preserve prior attempts'
    args.output.mkdir(parents=True)
    codec = ChatCodec(args.model, 'You are a friendly desktop companion. Reply briefly and naturally.')
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
    deployed = '/opt/opi-vlm/npu-kv-probe/' + args.tag
    remote = '/opt/opi-vlm/npu-kv-probe/' + args.output.name
    subprocess.run(ssh + ['mkdir ' + shlex.quote(remote)], check=True, timeout=20)
    for name in ('resident_session.c', 'full_model.c', 'board_probe.c', 'batch_prefill.h', 'greedy_i16.h'):
        subprocess.run(scp + [str(scripts / name), 'root@192.168.88.29:' + remote + '/'], check=True, timeout=60)
    build = (f'cd {remote} && cc -O2 -Wall -Wextra -Werror -DMODEL_DIM={manifest["dim"]} '
             f'-DMODEL_KV={manifest["kv_heads"]} -DMODEL_HD={manifest["head_dim"]} '
             f'-DROPE_BASE={manifest["rope_base"]} -DFUSED_LAYERS={runtime["fusion_width"]} '
             f'-DKEY_CACHE_TRANSPOSED={int(runtime["key_cache_transposed"])} '
             f'-DHEAD_SHARDS={len(manifest.get("head_shards", [{}]))} '
             f'-DHEAD_CHUNK={manifest.get("head_shards", [{"vocab_count": manifest["vocab_size"]}])[0]["vocab_count"]} '
             'resident_session.c -lNBGlinker -lVIPhal -lm -o resident')
    subprocess.run(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c ' + shlex.quote(build)],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, timeout=1900)
    worker_count = 0
    workers = []
    def factory():
        nonlocal worker_count
        worker_count += 1
        cmd = ['taskset', '-c', '6,7', remote + '/resident', deployed, str(runtime['decoder_layers']),
               str(runtime['capacity']), str(manifest['vocab_size']), deployed + '/embeddings.f32',
               str(runtime['prefill_batch_size']), str(manifest['eos_token_ids'][0])]
        # The remote FD3 carries private events through SSH stdout. SDK stdout
        # goes into its own board log, never into the protocol stream.
        shell = ('exec 3>&1; exec >>' + shlex.quote(remote + f'/worker{worker_count}.log') +
                 ' 2>&1; export OPI_SESSION_FD=3; exec ' + shlex.join(cmd))
        worker = ResidentWorker(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c ' + shlex.quote(shell)],
                                args.output, args.output / f'transport{worker_count}.log',
                                events_on_stdout=True)
        workers.append(worker)
        return worker
    engine = ChatEngine(codec, factory, args.model.stem, runtime['capacity'])
    server = make_server(engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result = {'status': 'running', 'scope': 'development HTTP/SSH multi-turn; not holdout or board-only performance',
              'model': str(args.model), 'reference': str(reference), 'layers': runtime['decoder_layers'],
              'capacity': runtime['capacity'], 'runs': [], 'independent_quality_accepted': False,
              'source_sha256': {n: hashlib.sha256((scripts / n).read_bytes()).hexdigest() for n in
                                ('resident_session.c', 'resident_worker.py', 'resident_http.py', 'run_resident_http_validation.py')}}
    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    def request(messages, max_new=48):
        start = time.monotonic()
        conn = http.client.HTTPConnection(*server.server_address, timeout=300)
        try:
            conn.request('POST', '/api/chat', json.dumps({'messages': messages, 'max_tokens': max_new}),
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            assert response.status == 200, response.read().decode()
            answer = ''
            metrics = None
            first = None
            done = False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                body = line[6:].strip()
                if body == b'[DONE]':
                    done = True
                    break
                event = json.loads(body)
                assert 'error' not in event, event
                text = event['choices'][0]['delta'].get('content', '')
                if text and first is None:
                    first = time.monotonic() - start
                answer += text
                if 'metrics' in event:
                    metrics = event['metrics']
            assert done and metrics is not None
            item = {'messages': messages, 'answer': answer, 'metrics': metrics,
                    'http_first_text_seconds': first, 'http_total_seconds': time.monotonic() - start,
                    'dropped_messages': int(response.getheader('X-Context-Dropped', '0'))}
            result['runs'].append(item)
            save()
            return item
        finally:
            conn.close()
    try:
        messages = []
        for text in ('I am making a cup of tea. Chat with me for a moment.',
                     'The rain outside sounds relaxing.', 'What were we talking about just now?'):
            messages.append({'role': 'user', 'content': text})
            reply = request(list(messages))
            assert reply['answer'], 'empty development response; retain evidence'
            messages.append({'role': 'assistant', 'content': reply['answer']})
        assert all(r['metrics']['reused_tokens'] > 0 for r in result['runs'][1:3]), 'template did not preserve KV'
        # Overflow drops complete old turns and forces the changed prefix to
        # reset, rather than silently retaining stale positions or KV entries.
        huge = [{'role': 'user', 'content': 'An old discussion. ' * 550},
                {'role': 'assistant', 'content': 'All right.'},
                {'role': 'user', 'content': 'Say hello in one short sentence.'}]
        cropped = request(huge, 24)
        assert cropped['dropped_messages'] == 2 and cropped['metrics']['reused_tokens'] == 0
        released = engine.close()
        assert released['driver_bytes'] == 0
        result['closed'] = released
        # The exact cropped request must produce the same full token sequence
        # after releasing/reloading every NPU buffer.
        restarted = request(huge[-1:], 24)
        assert restarted['metrics']['tokens'] == cropped['metrics']['tokens']
        result['reloaded_closed'] = engine.close()
        assert result['reloaded_closed']['driver_bytes'] == 0
        result.update(status='passed', worker_loads=worker_count,
                      checks=['three HTTP turns with KV reuse', 'whole-turn overflow and reset',
                              'release to zero driver bytes', 'reload gives identical complete token sequence'])
        save()
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        if engine.worker is not None:
            engine.worker.abort()
        for worker in workers:
            worker.abort()
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']), 'worker_loads': worker_count}))


if __name__ == '__main__':
    main()
