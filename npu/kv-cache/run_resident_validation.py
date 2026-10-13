"""Validate an isolated resident worker against an existing full-logits fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.tag.replace('-', '').isalnum()
    assert not args.output.exists(), 'retain existing evidence'
    reference = args.reference.resolve()
    source = reference.parents[1]
    manifest = json.loads((source / 'trained-manifest.json').read_text())
    runtime = json.loads((reference / 'tokens.json').read_text())
    assert runtime['decoder_layers'] == manifest['language_layers_total']
    assert runtime['fixed_feedback'] and runtime['prefill_batch_size']
    assert json.loads((reference / 'full-validation-w8.json').read_text())['execution_gate_v1']['passed']
    config = {'prompt': json.loads((reference / 'prompt.json').read_text())['token_ids'],
              'feedback': runtime['feedback_tokens'], 'cached_prefix': runtime['cached_prefix_tokens'],
              'expected_tokens': runtime['tokens'],
              'expected_sha256': hashlib.sha256((reference / 'logits.f32').read_bytes()).hexdigest(),
              'capacity': runtime['capacity'], 'layers': runtime['decoder_layers'],
              'vocab': manifest['vocab_size'], 'batch': runtime['prefill_batch_size'],
              'eos': manifest['eos_token_ids'][0]}
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    scripts = Path(__file__).resolve().parent
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
    deployed = '/opt/opi-vlm/npu-kv-probe/' + args.tag
    remote = '/opt/opi-vlm/npu-kv-probe/' + args.output.name
    assert args.output.name.replace('-', '').isalnum()
    subprocess.run(ssh + ['mkdir ' + shlex.quote(remote)], check=True, timeout=20)
    for name in ('resident_session.c', 'resident_worker.py', 'full_model.c', 'board_probe.c',
                 'batch_prefill.h', 'greedy_i16.h'):
        subprocess.run(scp + [str(scripts / name), 'root@192.168.88.29:' + remote + '/'],
                       check=True, timeout=60)
    build = (f'cd {remote} && cc -O2 -Wall -Wextra -Werror -DMODEL_DIM={manifest["dim"]} '
             f'-DMODEL_KV={manifest["kv_heads"]} -DMODEL_HD={manifest["head_dim"]} '
             f'-DROPE_BASE={manifest["rope_base"]} -DFUSED_LAYERS={runtime["fusion_width"]} '
             f'-DKEY_CACHE_TRANSPOSED={int(runtime["key_cache_transposed"])} '
             f'-DHEAD_SHARDS={len(manifest.get("head_shards",[{}]))} '
             f'-DHEAD_CHUNK={manifest.get("head_shards",[{"vocab_count":manifest["vocab_size"]}])[0]["vocab_count"]} '
             'resident_session.c -lNBGlinker -lVIPhal -lm -o resident')
    config.update(deployed=deployed, remote=remote)
    code = r'''
import hashlib,json,pathlib,time
from resident_worker import ResidentWorker
c=CONFIG
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
root=pathlib.Path(c['remote']);dump=root/'dumps';dump.mkdir()
deadline=time.monotonic()+300
while int(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text())>60000:
    if time.monotonic()>deadline:raise TimeoutError('resident validation cooling')
    time.sleep(1)
cmd=['taskset','-c','6,7',str(root/'resident'),c['deployed'],str(c['layers']),str(c['capacity']),
     str(c['vocab']),c['deployed']+'/embeddings.f32',str(c['batch']),str(c['eos'])]
w=ResidentWorker(cmd,root,root/'worker.log',dump)
result={'status':'running','scope':'diagnostic full-depth KV reuse, not performance or dialogue quality',
        'ready':w.ready,'runs':[],'config':c}
def save():
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def run(tokens,limit,feedback=None,reset=False,diagnostic=False):
    done=list(w.generate(tokens,limit,feedback,reset,diagnostic))[-1]
    result['runs'].append(done);save();return done
try:
    prime=run(c['prompt'][:c['cached_prefix']],1,reset=True)
    assert prime['kv_tokens']==c['cached_prefix']
    cached=run(c['prompt'],len(c['feedback']),c['feedback'],diagnostic=True)
    assert cached['reused_tokens']==c['cached_prefix']
    assert cached['tokens']==c['expected_tokens']
    assert sha(dump/'2-logits.f32')==c['expected_sha256'],'resident cached logits differ'
    fresh=run(c['prompt'],len(c['feedback']),c['feedback'],reset=True,diagnostic=True)
    assert fresh['reused_tokens']==0
    assert fresh['tokens']==c['expected_tokens']
    assert sha(dump/'3-logits.f32')==c['expected_sha256'],'reset logits differ'
    # A previous final predicted token has not yet entered KV. It must be
    # processed exactly once on the next request, alongside its new suffix.
    continued=c['prompt']+c['feedback']+[c['prompt'][-2],c['prompt'][-1]]
    next_turn=run(continued,4)
    assert next_turn['reused_tokens']==len(c['prompt'])+len(c['feedback'])-1
    assert next_turn['new_tokens']==3
    # Changing history must not retain any of that session's KV.
    changed=run(c['prompt'],1)
    assert changed['reused_tokens']==0
    result['closed']=w.close();assert result['closed']['driver_bytes']==0
    result['status']='passed';save()
except BaseException as exc:
    result.update(status='failed',error=str(exc));save();w.abort();raise
print(json.dumps(result))
'''.replace('CONFIG', repr(config), 1)
    command = build + ' && python3 -c ' + shlex.quote(code)
    args.output.mkdir(parents=True)
    with (args.output / 'run.log').open('x') as log:
        run = subprocess.run(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c ' +
                                    shlex.quote(command)], stdout=log, stderr=subprocess.STDOUT,
                             timeout=2400)
    subprocess.run(scp + ['root@192.168.88.29:' + remote + '/result.json', str(args.output / 'result.json')],
                   check=False, timeout=60)
    subprocess.run(scp + ['root@192.168.88.29:' + remote + '/worker.log', str(args.output / 'worker.log')],
                   check=False, timeout=60)
    run.check_returncode()
    result = json.loads((args.output / 'result.json').read_text())
    assert result['status'] == 'passed'
    result['source_sha256'] = {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                               for name in ('resident_session.c', 'resident_worker.py', 'full_model.c', 'batch_prefill.h')}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('resident all-layer cached/reset logits identical; last-token accounting passed')


if __name__ == '__main__':
    main()
