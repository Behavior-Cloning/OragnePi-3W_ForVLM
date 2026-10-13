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
p.add_argument('--group-head',action='append',default=[],help='submit all vocabulary shards as one NPU group for these labels')
p.add_argument('--init-buffer-flush',action='append',default=[],help='enable diagnostic buffer initialization flush for these labels')
p.add_argument('--host-output-buffers',action='append',default=[],help='use SDK host-backed output allocations for these labels')
p.add_argument('--prefill-host-hidden',action='append',default=[],help='host transfer between prefill blocks for these labels')
p.add_argument('--prefill-flush-output',action='append',default=[],help='flush hidden outputs before batched graph calls for these labels')
p.add_argument('--prefill-clear-output',action='append',default=[],help='clear and flush hidden output before batched graph calls for these labels')
p.add_argument('--prefill-scalar-block',nargs=2,action='append',default=[],metavar=('LABEL','BLOCK'))
p.add_argument('--prefill',nargs=2,action='append',default=[],metavar=('LABEL','TOKENS'))
p.add_argument('--prefill-scalar-prefix',nargs=2,action='append',default=[],metavar=('LABEL','TOKENS'),
    help='diagnostic preparation policy; retained and checked in every measured result')
p.add_argument('--cpu-affinity',help='optional taskset CPU list for the NPU host process, e.g. 6,7')
p.add_argument('--fixed-feedback',action='store_true',help='same frozen input trajectory, retaining each variant own predictions; not natural generation')
p.add_argument('--max-response-temp-mc',type=int,default=0,help='optional matched cooling with KV resident before the incremental response timer')
p.add_argument('--temperature-band-mc',type=int,nargs=2)
p.add_argument('--variant-reference',nargs=2,action='append',default=[],metavar=('LABEL','FOLDER'),
    help='with fixed feedback, independently validated diagnostic output for each variant')
a = p.parse_args()
from board_host_lock import lock_board_from_host
lock_board_from_host()
m = json.loads((a.source / 'trained-manifest.json').read_text())
r = json.loads((a.reference / 'tokens.json').read_text())
variants = []
assert 1 <= a.repeats <= 20
assert 35000 <= a.max_start_temp_mc <= 70000
assert a.max_response_temp_mc==0 or 35000<=a.max_response_temp_mc<=70000
if a.temperature_band_mc:
    lo,hi=a.temperature_band_mc
    assert 35000<=lo<hi<=80000 and hi-lo>=5000 and not a.max_response_temp_mc
if a.cpu_affinity:
    assert all(c in '01234567,-' for c in a.cpu_affinity) and len(a.cpu_affinity)<=24
for label, tag, case, group, fusion in a.variant:
    assert all(v.replace('-', '').isalnum() for v in (label, tag, case, a.experiment))
    group, fusion = int(group), int(fusion)
    assert fusion in (1, 2, 4) and len(m['layers']) % fusion == 0
    assert 0 <= group <= len(m['layers']) // fusion
    variants.append({'label': label, 'remote': '/opt/opi-vlm/npu-kv-probe/' + tag,
                     'case': 'board-' + case, 'group': group, 'fusion': fusion,'fast_greedy':label in a.fast_greedy,
                     'grouped_head':label in a.group_head,
                     'init_buffer_flush':label in a.init_buffer_flush,
                     'host_output_buffers':label in a.host_output_buffers,'prefill_host_hidden':label in a.prefill_host_hidden,
                     'prefill_flush_output':label in a.prefill_flush_output,
                     'prefill_clear_output':label in a.prefill_clear_output})
assert len({v['label'] for v in variants}) == len(variants)
assert set(a.fast_greedy).issubset({v['label'] for v in variants})
assert set(a.group_head).issubset({v['label'] for v in variants})
assert set(a.init_buffer_flush).issubset({v['label'] for v in variants})
assert set(a.host_output_buffers).issubset({v['label'] for v in variants})
assert set(a.prefill_host_hidden).issubset({v['label'] for v in variants})
assert set(a.prefill_flush_output).issubset({v['label'] for v in variants})
assert set(a.prefill_clear_output).issubset({v['label'] for v in variants})
references=dict(a.variant_reference)
prefill=dict(a.prefill)
assert len(prefill)==len(a.prefill) and all(2<=int(n)<=256 for n in prefill.values())
assert set(prefill).issubset({v['label'] for v in variants})
for v in variants:v['prefill_tokens']=int(prefill.get(v['label'],0))
scalar_blocks=dict(a.prefill_scalar_block)
assert len(scalar_blocks)==len(a.prefill_scalar_block) and set(scalar_blocks).issubset(prefill)
for v in variants:
    v['scalar_block']=int(scalar_blocks.get(v['label'],-1))
    assert -1<=v['scalar_block']<len(m['layers'])//v['fusion']
scalar_prefix=dict(a.prefill_scalar_prefix)
assert len(scalar_prefix)==len(a.prefill_scalar_prefix) and set(scalar_prefix).issubset(prefill)
assert all(0<=int(n)<=r['prompt_tokens'] for n in scalar_prefix.values())
for v in variants:v['scalar_prefix']=int(scalar_prefix.get(v['label'],0))
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
        assert expected['capacity']==r['capacity'] and expected.get('cached_prefix_tokens',0)==r.get('cached_prefix_tokens',0)
        assert expected.get('prefill_batch_size',0)==v['prefill_tokens']
        assert expected.get('prefill_scalar_prefix',0)==v['scalar_prefix']
        assert expected.get('init_buffer_flush',False)==v['init_buffer_flush']
        assert expected.get('host_output_buffers',False)==v['host_output_buffers']
        assert expected.get('prefill_host_hidden',False)==v['prefill_host_hidden']
        assert expected.get('prefill_flush_output',False)==v['prefill_flush_output']
        assert expected.get('prefill_clear_output',False)==v['prefill_clear_output']
        assert expected.get('prefill_scalar_block',-1)==v['scalar_block']
        assert not any(expected.get(k,False) for k in ('poison_prefill_output','capture_prefill','prefill_rebind_outputs'))
        assert not any(expected.get(k,False) for k in ('audit_outputs','capture_decode','decode_block_trace')), 'readback is diagnostic only'
        assert (folder/'prompt.i32').read_bytes()==(a.reference/'prompt.i32').read_bytes()
        assert validation['execution_gate_v1']['passed'] and validation['runtime']==expected
        v['expected_tokens']=expected['tokens']
        v['key_cache_transposed']=expected.get('key_cache_transposed',False)
        v['expected_logits_sha256']=hashlib.sha256((folder/'logits.f32').read_bytes()).hexdigest()
        v['validation']=validation['execution_gate_v1']
        v['feedback_sha256']=hashlib.sha256((folder/'feedback.i32').read_bytes()).hexdigest()
else:
    assert not r.get('fixed_feedback')
c = {'variants': variants, 'experiment': a.experiment, 'repeats': a.repeats,
     'fixed_feedback':a.fixed_feedback,'feedback_tokens':r.get('feedback_tokens',[]),
     'max_start_temperature_mC': a.max_start_temp_mc,
     'thermal_protocol': 'original-55C' if a.max_start_temp_mc == 55000 else 'supplementary-current-conditions',
     'layers': len(m['layers']), 'capacity': r['capacity'], 'vocab': m['vocab_size'],
     'cached_prefix':r.get('cached_prefix_tokens',0),
     'host_cpu_affinity':a.cpu_affinity,
     'eos': m['eos_token_ids'][0], 'tokens': r['tokens'], 'prompt_tokens': r['prompt_tokens'],
     'prompt_sha256': hashlib.sha256((a.reference / 'prompt.i32').read_bytes()).hexdigest(),
     'logits_sha256': hashlib.sha256((a.reference / 'logits.f32').read_bytes()).hexdigest()}
c['max_response_temperature_mC']=a.max_response_temp_mc
c['temperature_band_mC']=a.temperature_band_mc
sampler_source=Path(__file__).with_name('resource_sampler.py').read_text()
c['resource_sampler_sha256']=hashlib.sha256(sampler_source.encode()).hexdigest()
code = '''
import hashlib,json,os,pathlib,shutil,subprocess,threading,time,urllib.request
c=CONFIG
required=len(c['variants'])*len(c['tokens'])*c['vocab']*4+512*1024**2
available=shutil.disk_usage('/opt/opi-vlm/npu-kv-probe').free
assert available>=required, f'insufficient board space: {available} free, need {required} including diagnostic dumps and reserve'
provenance={}
def health():
    with urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=3) as response:
        value=json.load(response)
    assert value['ready'] and not value['busy'] and not value['loading'] and not value['maintenance'], 'production backend is active/unavailable'
    result={k:value[k] for k in ('model_id','backend','ready','busy','loading','maintenance','metrics')}
    result['frequencies']={p.name:{n:(p/n).read_text().strip() for n in ('scaling_cur_freq','scaling_max_freq','scaling_governor')}
        for p in pathlib.Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}
    result['npu_frequency_hz']=int(pathlib.Path('/sys/class/devfreq/3600000.npu/cur_freq').read_text())
    result['metrics']['web_cached_temperature_c']=result['metrics']['temperature_c']
    result['metrics']['temperature_c']=int(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
    result['temperature_sample_monotonic_seconds']=time.monotonic()
    result['temperature_source']='sysfs-direct'
    return result
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
    paths=list(root.glob('layer*.nb'))+list(root.glob('prefill-layer*.nb'))+[root/(v['case']+'-full_model')]
    if c['temperature_band_mC']:paths.extend([root/'thermal_condition.py',root/'thermal_heater'])
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
            env.pop('OPI_AUDIT_OUTPUTS',None);env.pop('OPI_INIT_BUFFER_FLUSH',None)
            env.pop('OPI_CAPTURE_PREFIX',None)
            for key in ('OPI_DECODE_CAPTURE_PREFIX','OPI_DECODE_CAPTURE_POSITION','OPI_DECODE_CAPTURE_BLOCK'):env.pop(key,None)
            env.pop('OPI_DECODE_BLOCK_TRACE',None)
            env.pop('OPI_HOST_OUTPUT_BUFFERS',None);env.pop('OPI_PREFILL_HOST_HIDDEN',None)
            for key in ('OPI_PREFILL_POISON_OUTPUT','OPI_PREFILL_REBIND_OUTPUTS','OPI_PREFILL_FLUSH_OUTPUT','OPI_PREFILL_CLEAR_OUTPUT','OPI_PREFILL_SCALAR_BLOCK'):env.pop(key,None)
            if v['scalar_block']>=0:env['OPI_PREFILL_SCALAR_BLOCK']=str(v['scalar_block'])
            if v['host_output_buffers']:env['OPI_HOST_OUTPUT_BUFFERS']='1'
            if v['prefill_host_hidden']:env['OPI_PREFILL_HOST_HIDDEN']='1'
            if v['prefill_flush_output']:env['OPI_PREFILL_FLUSH_OUTPUT']='1'
            if v['prefill_clear_output']:env['OPI_PREFILL_CLEAR_OUTPUT']='1'
            if v['init_buffer_flush']:env['OPI_INIT_BUFFER_FLUSH']='1'
            env.pop('OPI_PREFILL_TOKENS',None);env.pop('OPI_CACHED_PREFIX',None);env.pop('OPI_PREFILL_SCALAR_PREFIX',None)
            if v['prefill_tokens']:env['OPI_PREFILL_TOKENS']=str(v['prefill_tokens'])
            if v['scalar_prefix']:env['OPI_PREFILL_SCALAR_PREFIX']=str(v['scalar_prefix'])
            if c['cached_prefix']:env['OPI_CACHED_PREFIX']=str(c['cached_prefix'])
            if c['fixed_feedback']:env['OPI_FEEDBACK_FILE']=v['case']+'-feedback.i32'
            env['OPI_DECODER_GROUP']=str(v['group'])
            env.pop('OPI_FAST_GREEDY',None)
            if v['fast_greedy']:env['OPI_FAST_GREEDY']='1'
            env.pop('OPI_GROUP_HEAD',None)
            if v['grouped_head']:env['OPI_GROUP_HEAD']='1'
            env['OPI_MAX_START_TEMP_MC']=str(c['max_start_temperature_mC'])
            env.pop('OPI_MAX_RESPONSE_TEMP_MC',None)
            if c['max_response_temperature_mC']:env['OPI_MAX_RESPONSE_TEMP_MC']=str(c['max_response_temperature_mC'])
            env.pop('OPI_THERMAL_LOW_MC',None);env.pop('OPI_THERMAL_HIGH_MC',None)
            if c['temperature_band_mC']:
                env['OPI_THERMAL_LOW_MC'],env['OPI_THERMAL_HIGH_MC']=map(str,c['temperature_band_mC'])
            if phase!='diagnostic':env['OPI_PERFORMANCE']='1'
            command=['./'+v['case']+'-full_model','.',str(c['layers']),str(c['capacity']),str(c['vocab']),
                'embeddings.f32',v['case']+'-prompt.i32',str(len(c['tokens'])),str(folder/'tokens.json'),
                str(folder/'logits.f32'),str(c['eos'])]
            if c['host_cpu_affinity']:command=['taskset','-c',c['host_cpu_affinity']]+command
            resource_sampler=DirectSampler()
            observations=[{'elapsed':0.,'monotonic_seconds':time.monotonic(),'health':health(),
                           'resources':resource_sampler.snapshot()}];errors=[];stop=threading.Event();started=time.monotonic()
            def observe():
                while not stop.wait(1):
                    try:observations.append({'elapsed':time.monotonic()-started,'monotonic_seconds':time.monotonic(),'health':health(),
                                             'resources':resource_sampler.snapshot(child.pid)})
                    except Exception as exc:errors.append(str(exc))
            with (folder/'run.stdout').open('w') as out,(folder/'run.stderr').open('w') as err:
                with subprocess.Popen(command,env=env,stdout=out,stderr=err) as child:
                    observer=threading.Thread(target=observe);observer.start()
                    try:
                        if child.wait(timeout=600):raise RuntimeError('NPU runner failed; see run.stderr')
                    except BaseException:
                        child.kill();child.wait();raise
                    finally:stop.set();observer.join()
            observations.append({'elapsed':time.monotonic()-started,'monotonic_seconds':time.monotonic(),'health':health(),
                                 'resources':resource_sampler.snapshot()})
            assert not errors,errors
            r=json.loads((folder/'tokens.json').read_text())
            assert r['tokens']==v.get('expected_tokens',c['tokens']), 'prediction trajectory changed'
            assert r.get('fixed_feedback',False)==c['fixed_feedback']
            if c['fixed_feedback']:assert r['feedback_tokens']==c['feedback_tokens']
            assert r['prompt_tokens']==c['prompt_tokens'] and r['fusion_width']==v['fusion']
            assert r.get('prefill_batch_size',0)==v['prefill_tokens'] and r.get('cached_prefix_tokens',0)==c['cached_prefix']
            assert r.get('prefill_scalar_prefix',0)==v['scalar_prefix']
            assert r.get('init_buffer_flush',False)==v['init_buffer_flush']
            assert r.get('grouped_head',False)==v['grouped_head']
            assert r.get('host_output_buffers',False)==v['host_output_buffers']
            assert r.get('prefill_host_hidden',False)==v['prefill_host_hidden']
            assert r.get('prefill_flush_output',False)==v['prefill_flush_output']
            assert r.get('prefill_clear_output',False)==v['prefill_clear_output']
            assert r.get('prefill_scalar_block',-1)==v['scalar_block']
            assert not r.get('capture_decode',False)
            if 'key_cache_transposed' in v:
                assert r.get('key_cache_transposed',False)==v['key_cache_transposed']
            assert r['driver_after_release_bytes']==0
            if c['max_response_temperature_mC']:
                assert r['max_response_temperature_mC']==c['max_response_temperature_mC']
                assert 0<=r['response_start_temperature_mC']<=c['max_response_temperature_mC']+1000
            if c['temperature_band_mC']:
                lo,hi=c['temperature_band_mC']
                assert r['thermal_low_mC']==lo and r['thermal_high_mC']==hi
                assert lo<=r['response_start_temperature_mC']<=hi, 'response outside requested temperature band'
            record={'phase':phase,'repeat':repeat,'label':v['label'],'result':r,'system_samples':observations,
                    'resource_summary':response_summary([s['resources'] for s in observations],
                        r['response_start_monotonic_seconds'],r['response_start_monotonic_seconds']+r['total_seconds']),
                    'monitor_scope':'1-second direct proc/sys counters plus web readiness checks; boundary-overlapping intervals excluded'}
            if phase=='diagnostic':
                digest=digest_file(folder/'logits.f32')
                assert digest==v.get('expected_logits_sha256',c['logits_sha256']), 'complete logits changed'
                record['logits_sha256']=digest
            else:
                assert 0<=r['temperature_before_mC']<=c['max_start_temperature_mC']
                assert r['max_start_temperature_mC']==c['max_start_temperature_mC']
            print(json.dumps(record),flush=True)
'''.replace('CONFIG', repr(c), 1)
code = sampler_source + '\n' + code
compile(code, '<remote-benchmark>', 'exec')
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
