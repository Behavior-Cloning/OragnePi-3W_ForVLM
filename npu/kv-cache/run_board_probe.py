"""Deploy isolated probe files and retain raw board output. SSH keys or SSHPASS."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--capacity', type=int, required=True)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--layer0', type=Path, required=True)
p.add_argument('--layer1', type=Path, required=True)
p.add_argument('--host', default='root@192.168.88.29')
p.add_argument('--tag', required=True)
p.add_argument('--dim', type=int, default=64)
p.add_argument('--kv', type=int, default=2)
p.add_argument('--head-dim', type=int, default=16)
p.add_argument('--rope-base', type=float, default=10000)
p.add_argument('--repeats', type=int, default=5)
args = p.parse_args()
if not 1 <= args.repeats <= 20:
    raise ValueError('repeats must be 1..20')
if not args.tag.replace('-', '').replace('_', '').isalnum():
    raise ValueError('invalid tag')
remote = '/opt/opi-vlm/npu-kv-probe/' + args.tag
result_dir = args.fixture / args.tag
result_dir.mkdir(exist_ok=True)
prefix = [str(Path.home()/'.local/bin/sshpass'), '-e'] if 'SSHPASS' in os.environ else []
ssh = prefix + ['ssh', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', args.host]
scp = prefix + ['scp', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8']
subprocess.run(ssh + ['mkdir -p ' + shlex.quote(remote)], check=True, timeout=20)
for source, name in [(args.layer0, 'layer0.nb'), (args.layer1, 'layer1.nb'),
                     (args.fixture/'hidden.f32', 'hidden.f32'),
                     (args.fixture/'reference.f32', 'reference.f32'),
                     (Path(__file__).with_name('board_probe.c'), 'board_probe.c')]:
    subprocess.run(scp + [str(source), args.host+':'+remote+'/'+name], check=True, timeout=120)
tokens = (args.fixture/'hidden.f32').stat().st_size // (args.dim*4)
command = (f'cd {shlex.quote(remote)} && cc -O2 -Wall -Wextra -Werror '
           f'-DMODEL_DIM={args.dim} -DMODEL_KV={args.kv} -DMODEL_HD={args.head_dim} -DROPE_BASE={args.rope_base} '
           '-o board_probe board_probe.c -lNBGlinker -lVIPhal -lm && '
           f'for i in $(seq 1 {args.repeats}); do timeout 180 ./board_probe layer0.nb layer1.nb {args.capacity} {tokens} '
           'hidden.f32 reference.f32 output.f32 || exit $?; done')
command='flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)
result = subprocess.run(ssh + [command], capture_output=True, text=True, timeout=1800+200*args.repeats)
(result_dir/'run.log').write_text(result.stdout)
(result_dir/'run.stderr').write_text(result.stderr)
records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
summary = {'exit_code': result.returncode, 'tag': args.tag, 'dim': args.dim,
           'kv_heads': args.kv, 'head_dim': args.head_dim, 'measurements': records}
(result_dir/'board-result.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
if records:
    subprocess.run(scp + [args.host+':'+remote+'/output.f32', str(result_dir/'output.f32')],
                   check=True, timeout=30)
raise SystemExit(result.returncode)
