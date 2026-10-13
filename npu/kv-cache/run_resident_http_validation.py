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
    parser.add_argument('--browser', action='store_true', help='also click the actual page in isolated Chrome')
    parser.add_argument('--stability-seconds', type=int, default=0,
                        help='sustained natural multi-turn generation; at least 1800 seconds')
    parser.add_argument('--holdout-reference', type=Path,
                        help='frozen workstation reference; wait for preparation, then acceptance only')
    parser.add_argument('--retain-after-overflow', type=int,
                        help='optional soft prompt-token target after whole-turn eviction; default unchanged')
    parser.add_argument('--dialogue-turns', type=int, default=0,
                        help='separate bounded natural-dialogue experiment, not 30-minute acceptance')
    args = parser.parse_args()
    assert args.stability_seconds == 0 or args.stability_seconds >= 1800
    assert 0 <= args.dialogue_turns <= 100
    assert not args.holdout_reference or args.retain_after_overflow is None, 'freeze holdout context policy'
    scripts = Path(__file__).resolve().parent
    # Queue waits may span later edits. Python modules are already loaded;
    # preserve their launch bytes rather than hashing a newer file after lock.
    python_source_sha256 = {n: hashlib.sha256((scripts / n).read_bytes()).hexdigest() for n in
                           ('resident_worker.py', 'resident_http.py', 'run_resident_http_validation.py')}
    reference = args.reference.resolve()
    manifest = json.loads((reference.parents[1] / 'trained-manifest.json').read_text())
    runtime = json.loads((reference / 'tokens.json').read_text())
    assert runtime['decoder_layers'] == manifest['language_layers_total']
    assert json.loads((reference / 'full-validation-w8.json').read_text())['execution_gate_v1']['passed']
    assert args.tag.replace('-', '').isalnum() and args.output.name.replace('-', '').isalnum()
    assert not args.output.exists(), 'preserve prior attempts'
    args.output.mkdir(parents=True)
    quality=None
    if args.holdout_reference:
        deadline=time.monotonic()+14400
        while True:
            path=args.holdout_reference/'state.json'
            if path.exists():
                quality=json.loads(path.read_text())
                if quality['status']=='failed':
                    raise RuntimeError('holdout reference failed')
                if quality['status']=='complete-reference-only':
                    break
            if time.monotonic()>deadline:
                raise TimeoutError('holdout reference preparation')
            time.sleep(10)
        assert Path(quality['candidate']).resolve()==reference
        assert quality['candidate_execution_validation_sha256']==hashlib.sha256((reference/'full-validation-w8.json').read_bytes()).hexdigest()
        assert len(quality['cases'])==36
        suite_path=Path(quality.get('holdout_path',scripts/'quality-holdout-v1.json'))
        suite_raw=suite_path.read_bytes()
        assert hashlib.sha256(suite_raw).hexdigest()==quality['holdout_sha256'], 'holdout suite changed after reference freeze'
        suite=json.loads(suite_raw)
    codec = ChatCodec(args.model, 'You are a friendly desktop companion. Reply briefly and naturally.')
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
    deployed = '/opt/opi-vlm/npu-kv-probe/' + args.tag
    remote = '/opt/opi-vlm/npu-kv-probe/' + args.output.name
    subprocess.run(ssh + ['mkdir ' + shlex.quote(remote)], check=True, timeout=20)
    if quality:
        subprocess.run(ssh + ['mkdir '+shlex.quote(remote+'/dumps')],check=True,timeout=20)
        free=int(subprocess.check_output(ssh+['stat -f -c %a:%S '+shlex.quote(remote)],text=True).split(':')[0])
        unit=int(subprocess.check_output(ssh+['stat -f -c %S '+shlex.quote(remote)],text=True))
        required=sum(row['generated_count'] for row in quality['cases'])*manifest['vocab_size']*4+512*2**20
        assert free*unit>=required,'insufficient free space for complete holdout logits and reserve'
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
                 ' 2>&1; export OPI_SESSION_FD=3; '+
                 ('export OPI_SESSION_DUMP_DIR='+shlex.quote(remote+'/dumps')+'; ' if quality else '')+
                 'exec ' + shlex.join(cmd))
        worker = ResidentWorker(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c ' + shlex.quote(shell)],
                                args.output, args.output / f'transport{worker_count}.log',
                                events_on_stdout=True)
        workers.append(worker)
        return worker
    engine = ChatEngine(codec, factory, args.model.stem, runtime['capacity'], args.retain_after_overflow)
    server = make_server(engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result = {'status': 'running', 'scope': 'development HTTP/SSH multi-turn; not holdout or board-only performance',
              'model': str(args.model), 'reference': str(reference), 'layers': runtime['decoder_layers'],
              'capacity': runtime['capacity'], 'runs': [], 'independent_quality_accepted': False,
              'retain_after_overflow': args.retain_after_overflow,
              'source_sha256': {**python_source_sha256,
                  'resident_session.c': hashlib.sha256((scripts / 'resident_session.c').read_bytes()).hexdigest()},
              'provenance_scope': 'Python source snapshot at launch before lock; C snapshot at deployment'}
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
        if args.browser:
            from browser_resident_check import validate_browser
            result['browser'] = validate_browser('http://%s:%s/' % server.server_address, args.output, engine)
            result['browser_closed'] = engine.close()
            assert result['browser_closed']['driver_bytes'] == 0
        if args.stability_seconds or args.dialogue_turns:
            start=time.monotonic()
            first_index=len(result['runs'])
            base=json.loads((reference/'prompt.json').read_text())['messages']
            conversation=[m for m in base if m['role']!='system']
            # Work near the deployed context capacity with real complete chat
            # turns. Each reply is fed back; overflow is handled by the same HTTP
            # path as the page, never by silently discarding cached positions.
            result['stability']={'status':'running','requested_seconds':args.stability_seconds,
                                 'requested_turns':args.dialogue_turns,
                                 'first_run_index':first_index,'scope':'HTTP natural dialogue; host gateway, board resident NPU; not audio concurrency'}
            save()
            turn=0
            topics=('Tell me a little more about the quiet evening.',
                    'What gentle activity could we do next?',
                    'Continue our conversation in a few sentences.',
                    'Describe a cozy place where we could sit and talk.')
            while time.monotonic()-start<args.stability_seconds or turn<args.dialogue_turns:
                reply=request(list(conversation),128)
                dropped=reply['dropped_messages']
                if dropped:
                    conversation=conversation[dropped:]
                if not reply['answer']:
                    raise RuntimeError('empty answer during sustained dialogue')
                conversation += [{'role':'assistant','content':reply['answer']},
                                 {'role':'user','content':topics[turn%len(topics)]}]
                turn+=1
                result['stability'].update(elapsed_seconds=time.monotonic()-start,completed_turns=turn)
                save()
            released=engine.close()
            assert released['driver_bytes']==0
            result['stability'].update(status='complete',closed=released,
                                      elapsed_seconds=time.monotonic()-start,
                                      note='completed duration; inspect per-turn latency, RSS, driver and temperatures before accepting no regression')
            save()
        if quality:
            import numpy as np
            import torch
            torch.set_num_threads(2)
            result['holdout']={'status':'running','reference':str(args.holdout_reference),
                               'holdout_sha256':quality['holdout_sha256'],'cases':[],
                               'scope':'NPU execution plus requantization vs standard FP32 dequantized Q8; not original training BF16',
                               'human_quality_review':'pending','thresholds':quality['gate']}
            save()
            worker=factory()
            total_kl=total_nll_delta=total_tokens=0
            try:
                for row in quality['cases']:
                    done=list(worker.generate(row['prompt_tokens'],row['generated_count'],row['feedback_tokens'],
                                              reset=True,diagnostic=True))[-1]
                    folder=args.output/row['case']
                    folder.mkdir()
                    local=folder/'npu-logits.f32'
                    remote_logits=remote+f'/dumps/{done["id"]}-logits.f32'
                    subprocess.run(scp+['root@192.168.88.29:'+remote_logits,str(local)],check=True,timeout=180)
                    remote_digest=subprocess.check_output(ssh+['sha256sum '+shlex.quote(remote_logits)],text=True,timeout=30).split()[0]
                    assert hashlib.sha256(local.read_bytes()).hexdigest()==remote_digest
                    expected_path=args.holdout_reference/row['case']/'reference-logits.f32'
                    assert hashlib.sha256(expected_path.read_bytes()).hexdigest()==row['logits_sha256']
                    expected=torch.from_numpy(np.fromfile(expected_path,np.float32).reshape(-1,manifest['vocab_size']))
                    actual=torch.from_numpy(np.fromfile(local,np.float32).reshape(expected.shape))
                    assert torch.isfinite(actual).all() and len(actual)==row['generated_count']
                    assert actual.argmax(-1).tolist()==done['tokens']
                    p=torch.log_softmax(expected,dim=-1);q=torch.log_softmax(actual,dim=-1)
                    kl=(p.exp()*(p-q)).sum(-1)
                    idx=torch.arange(len(actual));targets=torch.tensor(row['feedback_tokens'])
                    delta=p[idx,targets]-q[idx,targets]
                    mean_kl=float(kl.mean());ratio=float(delta.mean().exp())
                    item={'case':row['case'],'token_count':len(actual),'mean_kl_nats':mean_kl,
                          'perplexity_ratio':ratio,'relative_l2':float((actual-expected).norm()/expected.norm()),
                          'argmax_fraction':float((actual.argmax(-1)==expected.argmax(-1)).float().mean()),
                          'numeric_passed':mean_kl<=.05 and ratio<=1.05,'runtime':done,
                          'npu_logits_sha256':remote_digest,'reference_logits_sha256':row['logits_sha256']}
                    (folder/'numeric.json').write_text(json.dumps(item,indent=2)+'\n')
                    result['holdout']['cases'].append(item)
                    total_kl+=float(kl.sum());total_nll_delta+=float(delta.sum());total_tokens+=len(actual)
                    save()
                closed=worker.close();assert closed['driver_bytes']==0
                # Independent natural generation follows the NPU's own history,
                # and is kept separately from fixed-trajectory numerical scores.
                natural=[]
                for case in suite['cases']:
                    messages=[]
                    for turn,text in enumerate(case['turns'],1):
                        messages.append({'role':'user','content':text})
                        reply=request(list(messages),96)
                        natural.append({'case':case['id']+f'-turn{turn}','answer':reply['answer'],
                                        'metrics':reply['metrics'],'checks':case['checks']})
                        # Preserve and report an empty response; don't turn it
                        # into a fabricated assistant turn or tune around it.
                        if not reply['answer']:
                            break
                        messages.append({'role':'assistant','content':reply['answer']})
                free_closed=engine.close();assert free_closed['driver_bytes']==0
                mean_kl=total_kl/total_tokens;ratio=float(np.exp(total_nll_delta/total_tokens))
                result['holdout'].update(status='complete-numeric-and-free-generation',
                    mean_kl_nats=mean_kl,perplexity_ratio=ratio,numeric_passed=mean_kl<=.05 and ratio<=1.05,
                    all_cases_numeric_passed=all(r['numeric_passed'] for r in result['holdout']['cases']),
                    natural=natural,closed=closed,free_closed=free_closed)
                save()
            finally:
                worker.abort()
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
