#!/usr/bin/env python3
"""Board-only companion runs: actual multi-turn replies, raw metrics, no automatic quality score."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import urllib.request
import benchmark_vlm as bench
import compare_vlm_models as runtime

ROOT = Path('/opt/opi-vlm')
LOCK = Path('/var/lib/opi-vlm-control/benchmark.lock')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['qwen3-17b-q4', 'qwen25-15b-q4'])
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output exists; choose a fresh path')
    plan = json.loads(args.plan.read_text())
    manifest = json.loads((ROOT/'vlm-models.json').read_text())
    for key in args.models:
        if key not in manifest:
            raise ValueError(key)
    settings = plan['settings']
    report = {'status': 'RUNNING', 'plan': plan, 'models': [], 'quality_scores': None,
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'notes': ['CPU A76 cores 6,7; 2 threads; no vision; actual assistant history.',
                        'Startup is page-cache-dependent, not a cold-start measurement.',
                        'No physical power measurement. Human quality review pending.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with LOCK.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        runtime.run('systemctl', 'stop', 'opi-vlm-model.service')
        for key in args.models:
            entry = {'id': key, 'manifest': manifest[key], 'conversations': []}
            report['models'].append(entry)
            save()
            try:
                directory = ROOT/'models' if key == 'smolvlm-256m' else ROOT/'models'/key
                files = [directory/f[0] for f in manifest[key]['files']]
                for file, (_, size, checksum) in zip(files, manifest[key]['files']):
                    if file.stat().st_size != size or runtime.digest(file) != checksum:
                        raise RuntimeError(f'Unverified model: {file.name}')
                runtime.stop()
                entry['cooldown_seconds'] = runtime.cooling()
                entry['startup'] = runtime.start(key, files, settings['context'])
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f'http://127.0.0.1:{runtime.PORT}/props') as response:
                    entry['props'] = json.load(response)
                for task in plan['tasks']:
                    conversation = {'id': task['id'], 'turns': [], 'valid': True}
                    entry['conversations'].append(conversation)
                    messages = [{'role': 'system', 'content': task['system_prompt']}]
                    for user in task['user_turns']:
                        messages.append({'role': 'user', 'content': user})
                        sampling = {k: settings[k] for k in ('temperature', 'top_p', 'top_k', 'repeat_penalty')}
                        sampling['seed'] = task['seed']
                        result = bench.infer(messages, max_tokens=settings['max_tokens'],
                                             port=runtime.PORT, model=key, sampling=sampling)
                        result['user'] = user
                        conversation['turns'].append(result)
                        if result.get('error') or result.get('finish_reason') != 'stop':
                            conversation['valid'] = False
                        if key == 'qwen3-17b-q4' and (result['reasoning'] or '<think>' in result['answer'] or '</think>' in result['answer']):
                            conversation['valid'] = False
                            result['nonthinking_violation'] = True
                        save()
                        print(json.dumps({'model': key, 'task': task['id'], 'turn': len(conversation['turns']),
                                          'answer': result['answer'], 'ttft': result['ttft_seconds'],
                                          'tps': result.get('timings', {}).get('predicted_per_second'),
                                          'error': result.get('error')}, ensure_ascii=False), flush=True)
                        if not conversation['valid']:
                            break
                        messages.append({'role': 'assistant', 'content': result['answer']})
            except Exception as error:
                entry['error'] = str(error)
                save()
            finally:
                runtime.stop()
        report['status'] = 'COMPLETE_WITH_ERRORS' if any(m.get('error') or any(not c['valid'] for c in m['conversations']) for m in report['models']) else 'COMPLETE'
    except BaseException:
        report['status'] = 'INTERRUPTED'
        raise
    finally:
        try:
            runtime.stop()
            runtime.run('systemctl', 'start', 'opi-vlm-model.service')
            bench.ready(port=8081, timeout=180)
            report['production_restored'] = True
        finally:
            LOCK.unlink(missing_ok=True)
            save()


if __name__ == '__main__':
    main()
