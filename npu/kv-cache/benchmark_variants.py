"""Alternate resident model variants, requiring identical complete outputs first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--variant', nargs=5, action='append', metavar=('LABEL', 'TAG', 'CASE', 'GROUP', 'FUSION'), required=True)
p.add_argument('--experiment', required=True)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--max-start-temp-mc', type=int, default=55000,
               help='explicit supplementary thermal condition; default preserves the original 55C protocol')
p.add_argument('--fast-greedy',action='append',default=[],help='enable exact integer argmax for these variant labels')
p.add_argument('--fixed-feedback',action='store_true',help='same frozen input trajectory, retaining each variant own predictions; not natural generation')
p.add_argument('--variant-reference',nargs=2,action='append',default=[],metavar=('LABEL','FOLDER'),
    help='with fixed feedback, independently validated diagnostic output for each variant')
a = p.parse_args()
m = json.loads((a.source / 'trained-manifest.json').read_text())
r = json.loads((a.reference / 'tokens.json').read_text())
variants = []
assert 1 <= a.repeats <= 20
assert 35000 <= a.max_start_temp_mc <= 70000
for label, tag, case, group, fusion in a.variant:
    assert all(v.replace('-', '').isalnum() for v in (label, tag, case, a.experiment))
    group, fusion = int(group), int(fusion)
    assert fusion in (1, 2, 4) and len(m['layers']) % fusion == 0
    assert 0 <= group <= len(m['layers']) // fusion
    variants.append({'label': label, 'remote': '/opt/opi-vlm/npu-kv-probe/' + tag,
                     'case': 'board-' + case, 'group': group, 'fusion': fusion,'fast_greedy':label in a.fast_greedy})
assert len({v['label'] for v in variants}) == len(variants)
assert set(a.fast_greedy).issubset({v['label'] for v in variants})
references=dict(a.variant_reference)
assert len(references)==len(a.variant_reference)
assert a.fixed_feedback == bool(references)
if a.fixed_feedback:
    assert r.get('fixed_feedback') and len(r['tokens'])==len(r['feedback_tokens'])
    assert set(references)=={v['label'] for v in variants}
    for v in variants:
        folder=Path(references[v['label']]);expected=json.loads((folder/'tokens.json').read_text())
        validation=json.loads((folder/'full-validation-w8.json').read_text())
        assert expected.get('fixed_feedback') and expected['feedback_tokens']==r['feedback_tokens']
        assert expected['prompt_tokens']==r['prompt_tokens'] and len(expected['tokens'])==len(r['tokens'])
        assert (folder/'prompt.i32').read_bytes()==(a.reference/'prompt.i32').read_bytes()
        assert validation['execution_gate_v1']['passed'] and validation['runtime']==expected
        v['expected_tokens']=expected['tokens']
        v['expected_logits_sha256']=hashlib.sha256((folder/'logits.f32').read_bytes()).hexdigest()
        v['validation']=validation['execution_gate_v1']
        v['feedback_sha256']=hashlib.sha256((folder/'feedback.i32').read_bytes()).hexdigest()
else:
    assert not r.get('fixed_feedback')
c = {'variants': variants, 'experiment': a.experiment, 'repeats': a.repeats,
     'fixed_feedback':a.fixed_feedback,'feedback_tokens':r.get('feedback_tokens',[]),
     'max_start_temperature_mC': a.max_start_temp_mc,
     'thermal_protocol': 'original-55C' if a.max_start_temp_mc == 55000 else 'supplementary-current-conditions',
     'layers': len(m['layers']), 'capacity': m['capacity'], 'vocab': m['vocab_size'],
     'eos': m['eos_token_ids'][0], 'tokens': r['tokens'], 'prompt_tokens': r['prompt_tokens'],
     'prompt_sha256': hashlib.sha256((a.reference / 'prompt.i32').read_bytes()).hexdigest(),
     'logits_sha256': hashlib.sha256((a.reference / 'logits.f32').read_bytes()).hexdigest()}
code = '''
import hashlib,json,os,pathlib,subprocess,threading,time,urllib.request
c=CONFIG
provenance={}
def health():
    with urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=3) as response:
        value=json.load(response)
    assert value['ready'] and not value['busy'] and not value['loading'] and not value['maintenance'], 'production backend is active/unavailable'
    return {k:value[k] for k in ('model_id','backend','ready','busy','loading','maintenance','metrics')}
def digest_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
for v in c['variants']:
    root=pathlib.Path(v['remote'])
    assert hashlib.sha256((root/(v['case']+'-prompt.i32')).read_bytes()).hexdigest()==c['prompt_sha256']
    if c['fixed_feedback']:
        assert digest_file(root/(v['case']+'-feedback.i32'))==v['feedback_sha256']
    paths=list(root.glob('layer*.nb'))+[root/(v['case']+'-full_model')]
    provenance[v['label']]={p.name:digest_file(p) for p in paths}
print(json.dumps({'provenance':provenance}),flush=True)
for phase in ('diagnostic','warmup','measure'):
    for repeat in range(c['repeats'] if phase=='measure' else 1):
        variants=c['variants'] if repeat%2==0 else list(reversed(c['variants']))
        for v in variants:
            os.chdir(v['remote'])
            folder=pathlib.Path(c['experiment']+'-'+v['label']+f'-{phase}-{repeat}')
            folder.mkdir(exist_ok=False)
            env=os.environ.copy()
            env.pop('OPI_FULL_TRACE',None);env.pop('OPI_PERFORMANCE',None)
            env.pop('OPI_FEEDBACK_FILE',None)
            if c['fixed_feedback']:env['OPI_FEEDBACK_FILE']=v['case']+'-feedback.i32'
            env['OPI_DECODER_GROUP']=str(v['group'])
            env.pop('OPI_FAST_GREEDY',None)
            if v['fast_greedy']:env['OPI_FAST_GREEDY']='1'
            env['OPI_MAX_START_TEMP_MC']=str(c['max_start_temperature_mC'])
            if phase!='diagnostic':env['OPI_PERFORMANCE']='1'
            command=['./'+v['case']+'-full_model','.',str(c['layers']),str(c['capacity']),str(c['vocab']),
                'embeddings.f32',v['case']+'-prompt.i32',str(len(c['tokens'])),str(folder/'tokens.json'),
                str(folder/'logits.f32'),str(c['eos'])]
            observations=[{'elapsed':0.,'health':health()}];errors=[];stop=threading.Event();started=time.monotonic()
            def observe():
                while not stop.wait(1):
                    try:observations.append({'elapsed':time.monotonic()-started,'health':health()})
                    except Exception as exc:errors.append(str(exc))
            observer=threading.Thread(target=observe);observer.start()
            with (folder/'run.stdout').open('w') as out,(folder/'run.stderr').open('w') as err:
                try:subprocess.run(command,env=env,stdout=out,stderr=err,check=True,timeout=600)
                finally:stop.set();observer.join()
            observations.append({'elapsed':time.monotonic()-started,'health':health()})
            assert not errors,errors
            r=json.loads((folder/'tokens.json').read_text())
            assert r['tokens']==v.get('expected_tokens',c['tokens']), 'prediction trajectory changed'
            assert r.get('fixed_feedback',False)==c['fixed_feedback']
            if c['fixed_feedback']:assert r['feedback_tokens']==c['feedback_tokens']
            assert r['prompt_tokens']==c['prompt_tokens'] and r['fusion_width']==v['fusion']
            assert r['driver_after_release_bytes']==0
            record={'phase':phase,'repeat':repeat,'label':v['label'],'result':r,'system_samples':observations,
                    'monitor_scope':'1-second health snapshots; short concurrent requests between samples may be missed'}
            if phase=='diagnostic':
                digest=digest_file(folder/'logits.f32')
                assert digest==v.get('expected_logits_sha256',c['logits_sha256']), 'complete logits changed'
                record['logits_sha256']=digest
            else:
                assert 0<=r['temperature_before_mC']<=c['max_start_temperature_mC']
                assert r['max_start_temperature_mC']==c['max_start_temperature_mC']
            print(json.dumps(record),flush=True)
'''.replace('CONFIG', repr(c), 1)
prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
ssh = prefix + ['ssh', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
report = {'config': c, 'status': 'running', 'runs': []}
a.output.parent.mkdir(parents=True, exist_ok=True)
def save():
    a.output.write_text(json.dumps(report, indent=2) + '\n')
save()
try:
    with subprocess.Popen(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock python3 -c ' + shlex.quote(code)],
                          stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            record = json.loads(line)
            if 'provenance' in record:
                report.update(record)
            else:
                report['runs'].append(record)
                print(record['phase'], record['repeat'], record['label'], record['result']['decode_tokens_per_second'], flush=True)
            save()
        if proc.wait():
            raise RuntimeError('board experiment failed; retained logs on board')
    metrics = ('decode_tokens_per_second', 'ttft_seconds', 'total_seconds', 'cpu_seconds', 'driver_allocated_bytes',
               'decode_decoder_api_seconds', 'decode_head_api_seconds', 'max_rss_kib', 'load_seconds')
    report['medians'] = {v['label']: {k: statistics.median(r['result'][k] for r in report['runs']
        if r['phase'] == 'measure' and r['label'] == v['label']) for k in metrics} for v in variants}
    report['status'] = 'complete'
except BaseException as exc:
    report.update(status='failed', error=str(exc))
    raise
finally:
    save()
print(json.dumps(report['medians'], indent=2))
