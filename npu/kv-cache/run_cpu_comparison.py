"""Sequential board CPU baselines. Requires isolated board venv with numpy/ORT."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,required=True)
p.add_argument('--host',default='root@192.168.88.29');args=p.parse_args()
root=Path(__file__).resolve().parents[2]
folder=root/f'npu/results/trained-smol-c{args.capacity}'
remote=f'/opt/opi-vlm/npu-kv-probe/cpu-smol-c{args.capacity}'
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8',args.host]
scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=20)
files=[folder/name for name in ['layer0.onnx','layer1.onnx','hidden.f32','reference.f32','trained-manifest.json']]
subprocess.run(scp+[str(v) for v in files]+[str(Path(__file__).with_name('cpu_board_probe.py')),args.host+':'+remote+'/'],
               check=True,timeout=120)
for threads,affinity,suffix in [(1,'',''),(2,'',''),(4,'',''),(1,'taskset -c 6 ', '-a76')]:
    command=f'{affinity}/opt/opi-vlm/npu-kv-probe/venv/bin/python {remote}/cpu_board_probe.py --folder {remote} --threads {threads}'
    result=subprocess.run(ssh+[command],capture_output=True,text=True,timeout=180)
    (folder/f'cpu-{threads}-threads{suffix}.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise SystemExit(result.stderr)
    subprocess.run(scp+[args.host+':'+remote+f'/cpu-{threads}-threads.json',str(folder/f'cpu-{threads}-threads{suffix}.json')],
                   check=True,timeout=20)
    print(f'capacity={args.capacity} threads={threads}{suffix} PASS',flush=True)
