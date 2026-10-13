"""Build pinned whisper.cpp on the board and validate a real ASR invocation.

This is preparation, not an audio-concurrency result. Public 11 s sample;
no microphone, echo cancellation, TTS or Chinese accuracy claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from board_host_lock import lock_board_from_host


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    bundle = root / 'npu/results/audio-asr-20261013/whisper-v1.8.0.tar.gz'
    model = root / 'models/whisper-tiny/ggml-tiny-q5_1.bin'
    sha = '818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7'
    state = {'status': 'waiting-model-download', 'source_commit': '41fc9dea6a4fe056424be86f61164413903fcff4',
             'model_source': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny-q5_1.bin',
             'model_sha256': sha, 'source_archive_sha256': digest(bundle),
             'sample': 'whisper.cpp samples/jfk.wav', 'sample_seconds': 11,
             'scope': 'real ASR setup and smoke only; not concurrency, Chinese accuracy or microphone pipeline'}
    def save():
        tmp = a.output / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(a.output / 'state.json')
    save()
    deadline = time.monotonic() + 3600
    while not model.exists() or model.stat().st_size != 32152673:
        assert time.monotonic() < deadline, 'ASR weight download not complete'
        time.sleep(10)
    assert digest(model) == sha
    state['status'] = 'waiting-board'
    save()
    lock_board_from_host()
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
    assert a.output.name.replace('-', '').isalnum()
    remote = '/opt/opi-vlm/npu-kv-probe/' + a.output.name
    state['remote'] = remote
    try:
        subprocess.run(ssh + ['mkdir ' + shlex.quote(remote)], check=True, timeout=30)
        subprocess.run(scp + [str(bundle), str(model), 'root@192.168.88.29:' + remote + '/'], check=True, timeout=300)
        code = '''
import hashlib,json,pathlib,subprocess,time
root=pathlib.Path(REMOTE)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'ggml-tiny-q5_1.bin')==MODEL_SHA
assert sha(root/'whisper-v1.8.0.tar.gz')==ARCHIVE_SHA
subprocess.run(['tar','-xzf',str(root/'whisper-v1.8.0.tar.gz'),'-C',str(root)],check=True)
src=root/'whisper.cpp';build=root/'build'
with (root/'build.log').open('x') as log:
 subprocess.run(['cmake','-S',str(src),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
  '-DBUILD_SHARED_LIBS=OFF','-DWHISPER_BUILD_TESTS=OFF','-DWHISPER_BUILD_SERVER=OFF',
  '-DWHISPER_CURL=OFF','-DGGML_NATIVE=ON'],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['nice','-n','10','taskset','-c','0,1','cmake','--build',str(build),
  '--target','whisper-cli','-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
exe=build/'bin/whisper-cli';wav=src/'samples/jfk.wav'
assert sha(wav)=='59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'
start=time.monotonic()
command=['taskset','-c','6,7',str(exe),'-m',str(root/'ggml-tiny-q5_1.bin'),
 '-f',str(wav),'-t','2','-l','en','-nt','-oj','-of',str(root/'smoke')]
with (root/'smoke.stdout').open('x') as out,(root/'smoke.stderr').open('x') as err:
 subprocess.run(command,stdout=out,stderr=err,check=True,timeout=180)
result=json.loads((root/'smoke.json').read_text())
answer=' '.join(x['text'] for x in result['transcription'])
assert 'country' in answer.lower(), 'ASR smoke did not decode expected sample content'
report={'status':'passed-setup-only','command':command,'elapsed_seconds':time.monotonic()-start,
 'transcript':answer,'binary_sha256':sha(exe),'model_sha256':MODEL_SHA,
 'sample_sha256':sha(wav),'sample_seconds':11,'whisper':result}
(root/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
'''
        code = code.replace('REMOTE', repr(remote)).replace('MODEL_SHA', repr(sha)).replace('ARCHIVE_SHA', repr(state['source_archive_sha256']))
        compile(code, '<board-asr-setup>', 'exec')
        state['status'] = 'building-and-smoke'
        save()
        with (a.output / 'run.log').open('x') as log:
            proc = subprocess.run(ssh + ['flock -w 1800 /run/opi-npu-kv-benchmark.lock python3 -c ' + shlex.quote(code)],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=1500)
        for name in ('build.log', 'smoke.json', 'smoke.stdout', 'smoke.stderr', 'result.json'):
            subprocess.run(scp + ['root@192.168.88.29:' + remote + '/' + name, str(a.output / name)], timeout=60)
        proc.check_returncode()
        state.update(status='passed-setup-only', result=json.loads((a.output / 'result.json').read_text()))
        save()
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
