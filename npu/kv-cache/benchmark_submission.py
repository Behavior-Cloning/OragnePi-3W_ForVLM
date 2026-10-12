"""Compare submission groups under one board lock, preserving weights and tokens.

Deploy the matching full_model runner first with run_full_model.py --deploy-only.
The reference must contain diagnostic logits, tokens and prompt for that variant.
"""
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
p.add_argument('--tag', required=True)
p.add_argument('--runner-case', required=True)
p.add_argument('--groups', nargs='+', type=int, required=True)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
for value in (args.tag, args.runner_case):
    if not value.replace('-', '').isalnum():
        raise ValueError('unsafe tag/case')
m = json.loads((args.source / 'trained-manifest.json').read_text())
if not 1 <= args.repeats <= 20 or not args.groups or len(set(args.groups)) != len(args.groups):
    raise ValueError('invalid repeats/groups')
if any(not 0 <= g <= len(m['layers']) for g in args.groups):
    raise ValueError('invalid group size')
reference = json.loads((args.reference / 'tokens.json').read_text())
config = {
    'remote': '/opt/opi-vlm/npu-kv-probe/' + args.tag,
    'case': 'board-' + args.runner_case,
    'groups': args.groups, 'repeats': args.repeats,
    'layers': len(m['layers']), 'capacity': m['capacity'], 'vocab': m['vocab_size'],
    'max_new': reference['generated_tokens'], 'eos': m['eos_token_ids'][0],
    'tokens': reference['tokens'], 'prompt_tokens': reference['prompt_tokens'],
    'prompt_sha256': hashlib.sha256((args.reference / 'prompt.i32').read_bytes()).hexdigest(),
    'logits_sha256': hashlib.sha256((args.reference / 'logits.f32').read_bytes()).hexdigest(),
}
remote_code = '''
import hashlib,json,os,pathlib,subprocess
c=CONFIG
os.chdir(c['remote'])
assert hashlib.sha256(pathlib.Path(c['case']+'-prompt.i32').read_bytes()).hexdigest()==c['prompt_sha256']
for phase in ('diagnostic','warmup','measure'):
    for repeat in range(c['repeats'] if phase=='measure' else 1):
        groups=c['groups'] if repeat%2==0 else list(reversed(c['groups']))
        for group in groups:
            folder=pathlib.Path(c['case']+f'-g{group}-{phase}-{repeat}')
            folder.mkdir(exist_ok=False)
            env=os.environ.copy()
            env.pop('OPI_FULL_TRACE',None)
            env.pop('OPI_PERFORMANCE',None)
            env['OPI_DECODER_GROUP']=str(group)
            if phase!='diagnostic':env['OPI_PERFORMANCE']='1'
            command=['./'+c['case']+'-full_model','.',str(c['layers']),str(c['capacity']),str(c['vocab']),
                'embeddings.f32',c['case']+'-prompt.i32',str(c['max_new']),str(folder/'tokens.json'),
                str(folder/'logits.f32'),str(c['eos'])]
            with (folder/'run.stdout').open('w') as out,(folder/'run.stderr').open('w') as err:
                subprocess.run(command,env=env,stdout=out,stderr=err,check=True,timeout=600)
            r=json.loads((folder/'tokens.json').read_text())
            assert r['tokens']==c['tokens'], 'token trajectory changed'
            assert r['prompt_tokens']==c['prompt_tokens'], 'prompt length changed'
            assert r['driver_after_release_bytes']==0, 'driver memory not released'
            record={'phase':phase,'repeat':repeat,'group':group,'result':r}
            if phase=='diagnostic':
                digest=hashlib.sha256((folder/'logits.f32').read_bytes()).hexdigest()
                assert digest==c['logits_sha256'], 'complete logits changed'
                record['logits_sha256']=digest
            else:assert 0<=r['temperature_before_mC']<=55000
            print(json.dumps(record),flush=True)
'''.replace('CONFIG', repr(config), 1)
prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
ssh = prefix + ['ssh', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
command = 'flock -w 1800 /run/opi-npu-kv-benchmark.lock python3 -c ' + shlex.quote(remote_code)
report = {'config': config, 'status': 'running', 'runs': [],
          'scope': 'full language chain, identical prompt and token trajectory; diagnostic logits must match byte-for-byte'}
args.output.parent.mkdir(parents=True, exist_ok=True)
def save():
    args.output.write_text(json.dumps(report, indent=2) + '\n')
save()
try:
    with subprocess.Popen(ssh + [command], text=True, stdout=subprocess.PIPE) as proc:
        for line in proc.stdout:
            r = json.loads(line)
            report['runs'].append(r)
            save()
            print(r['phase'], r['repeat'], r['group'], r['result']['decode_tokens_per_second'], flush=True)
        if proc.wait():
            raise RuntimeError('board submission experiment failed; inspect retained board logs')
    keys = ('decode_tokens_per_second', 'ttft_seconds', 'total_seconds', 'cpu_seconds',
            'driver_allocated_bytes', 'decode_decoder_api_seconds', 'decode_head_api_seconds',
            'npu_api_seconds', 'max_rss_kib')
    report['medians'] = {str(g): {k: statistics.median(r['result'][k] for r in report['runs']
        if r['group'] == g and r['phase'] == 'measure') for k in keys} for g in args.groups}
    report['status'] = 'complete'
except BaseException as exc:
    report['status'] = 'failed'
    report['error'] = str(exc)
    raise
finally:
    save()
print(json.dumps(report['medians'], indent=2))
