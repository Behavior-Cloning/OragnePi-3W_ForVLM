"""Copy exact failed-run logs and board health under both experiment locks."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--phase', choices=('diagnostic', 'warmup', 'measure'), required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    raw = json.loads(a.result.read_text())
    assert raw['status'] == 'failed'
    c = raw['config']
    v = next(v for v in c['variants'] if v['label'] == a.label)
    remote = str(Path(v['remote']) / (c['experiment'] + '-' + a.label + f'-{a.phase}-{a.repeat}'))
    assert remote.startswith('/opt/opi-vlm/npu-kv-probe/')
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    code = r'''
import json,pathlib,shutil,subprocess,urllib.request
p=pathlib.Path(REMOTE)
d={'remote':str(p),'files':{},'free_bytes':shutil.disk_usage('/').free,
   'temperature_mC':int(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text()),
   'meminfo':pathlib.Path('/proc/meminfo').read_text(),
   'kernel_tail':subprocess.run(['dmesg'],capture_output=True,text=True).stdout[-20000:]}
for name in ('run.stderr','run.stdout','tokens.json'):
    f=p/name
    d['files'][name]=f.read_text(errors='replace')[-64000:] if f.exists() else None
try:
    with urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=5) as response:
        d['web_health']=json.load(response)
except Exception as exc:d['web_error']=str(exc)
print(json.dumps(d))
'''.replace('REMOTE', repr(remote), 1)
    prefix = [str(Path.home() / '.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
    ssh = prefix + ['ssh', '-T', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', 'root@192.168.88.29']
    output = subprocess.check_output(ssh + ['flock -w 120 /run/opi-npu-kv-benchmark.lock python3 -c ' + shlex.quote(code)],
                                     text=True, timeout=160)
    data = json.loads(output)
    a.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: data[k] for k in ('remote', 'free_bytes', 'temperature_mC')}))
    print(data['files']['run.stderr'])


if __name__ == '__main__':
    main()
