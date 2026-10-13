"""Resume full-depth candidate checks and five-run measurements on deployed graphs.

The request file is immutable once work starts. Each failed case is retained;
independent cases can continue. No score here accepts the held-out dialogue set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from attempt_paths import select_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    project = scripts.parents[1]
    request_bytes = args.requests.read_bytes()
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    requests = json.loads(request_bytes)
    assert len({r['case'] for r in requests}) == len(requests)
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'requests_sha256': request_hash, 'cases': {},
        'independent_quality_accepted': False,
        'scope': 'full-depth development trajectories; response-start temperature bands'}
    assert state['requests_sha256'] == request_hash, 'use a new output for changed requests'

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    def run(case, step, command):
        with (args.output / (case + '-' + step + '.log')).open('x') as log:
            subprocess.run([sys.executable, str(scripts / command[0]), *map(str, command[1:])],
                           cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)

    state['status'] = 'running'
    save()
    for request in requests:
        case = request['case']
        row = state['cases'].setdefault(case, {'status': 'pending', 'config': request})
        if row['status'] in ('complete', 'quality-failed') or (row['status']=='failed' and not args.retry_failed):
            continue
        try:
            source = project / request['source']
            reference = source / request['dtype'] / ('board-' + case)
            row.update(status='running', active='wait-feedback', reference=str(reference))
            save()
            feedback = project / request['feedback']
            deadline = time.monotonic() + 14400
            while not feedback.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('CPU feedback missing: ' + str(feedback))
                time.sleep(15)
            if request.get('wait_reference'):
                row['active']='waiting-existing-full-run';save()
                expected_bytes=128*json.loads((source/'trained-manifest.json').read_text())['vocab_size']*4
                # The producer owns validation too. Waiting only for tokens
                # allowed a second runner to overwrite its validation files.
                def reference_ready():
                    try:
                        if (reference/'logits.f32').stat().st_size != expected_bytes:
                            return False
                        for name in ('tokens.json', 'full-validation-w8.json', 'full-validation.json'):
                            json.loads((reference/name).read_text())
                        return True
                    except (FileNotFoundError, json.JSONDecodeError):
                        return False
                while not reference_ready():
                    if (reference/'exit.json').exists():
                        assert json.loads((reference/'exit.json').read_text())['exit_code']==0, 'dependent full run failed'
                    if time.monotonic()>deadline:
                        raise TimeoutError('existing full run not ready: '+str(reference))
                    time.sleep(15)
            if not (reference / 'tokens.json').exists():
                row['active'] = 'full-depth-run'
                save()
                command = ['run_full_model.py', '--source', source, '--dtype', request['dtype'],
                           '--tag', request['tag'], '--case', case, '--prompt-folder',
                           project / request['fixture'], '--max-new', '128', '--cached-prefix',
                           request['history'], '--decoder-group', request['group'],
                           '--fusion-width', '2', '--prefill-variant', request['prefill'],
                           '--fast-greedy', '--cpu-affinity', '6,7', '--skip-deploy',
                           '--feedback-from', feedback]
                if request.get('group_head'):
                    command.append('--group-head')
                run(case, 'full', command)
            runtime = json.loads((reference / 'tokens.json').read_text())
            manifest = json.loads((source / 'trained-manifest.json').read_text())
            assert runtime['decoder_layers'] == manifest['language_layers_total']
            assert runtime['head_on_npu'] and runtime['fixed_feedback']
            assert runtime['generated_tokens'] == 128
            assert runtime['cached_prefix_tokens'] == request['history']
            row['active'] = 'full-logits-validation'
            save()
            for weights, suffix in [('exported', '-w8'), ('original', '')]:
                if not (reference / ('full-validation' + suffix + '.json')).exists():
                    run(case, weights, ['validate_full.py', '--source', source,
                        '--model', project / request['model'], '--dtype', request['dtype'],
                        '--case', case, '--reference-weights', weights])
            validation = json.loads((reference / 'full-validation-w8.json').read_text())
            original = json.loads((reference / 'full-validation.json').read_text())
            row.update(execution_gate=validation['execution_gate_v1'],
                       original_q8_development={
                           'relative_l2': original['relative_l2_all_logits'],
                           'mincos': original['min_step_cosine'], 'kl': original['mean_kl_nats'],
                           'ppl_ratio': original['fixed_trajectory_nll']['perplexity_ratio']})
            if (not validation['execution_gate_v1']['passed'] or
                    original['mean_kl_nats'] > .05 or
                    original['fixed_trajectory_nll']['perplexity_ratio'] > 1.05):
                row.update(status='quality-failed', active=None)
                save()
                continue
            for band in request.get('bands', ['medium']):
                assert band in ('low', 'medium')
                row['active'] = 'five-repeat-' + band
                save()
                output = args.output / (case + '-' + band + '.json')
                output,_,attempt=select_attempt(output,None,args.output/(case+'-'+band+'.log'),args.retry_failed)
                attempt_suffix='' if attempt==1 else f'-attempt{attempt}'
                if not output.exists():
                    low, high = (40000, 50000) if band == 'low' else (55000, 64999)
                    command = ['benchmark_variants.py', '--source', source, '--reference', reference,
                               '--variant', 'candidate', request['tag'], case, request['group'], '2',
                               '--variant-reference', 'candidate', reference,
                               '--prefill', 'candidate', runtime['prefill_batch_size'],
                               '--fast-greedy', 'candidate', '--fixed-feedback',
                               '--cpu-affinity', '6,7', '--max-start-temp-mc', '68000',
                               '--temperature-band-mc', low, high, '--repeats', '5',
                               '--experiment', args.output.name + '-' + case + '-' + band + attempt_suffix,
                               '--output', output]
                    if request.get('group_head'):
                        command += ['--group-head', 'candidate']
                    run(case, band+attempt_suffix, command)
                result = json.loads(output.read_text())
                assert result['status'] == 'complete', 'retain partial measurement; do not overwrite'
                row.setdefault('bands', {})[band] = {
                    'result': str(output), 'medians': result['medians']['candidate']}
                save()
            row.update(status='complete', active=None)
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            state['status'] = 'failed'
            save()
            raise  # Connection/storage/runtime faults must stop board work.
        save()
    state['status'] = ('complete' if all(r['status'] == 'complete' for r in state['cases'].values())
                       else 'complete-with-rejected-cases')
    save()


if __name__ == '__main__':
    main()
