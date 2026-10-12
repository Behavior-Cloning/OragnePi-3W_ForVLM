"""Deploy compiled activation probes and retain board outputs for CPU checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
folder=args.folder;scripts=Path(__file__).resolve().parent
records=json.loads((folder/'activation-manifest.json').read_text())
remote='/opt/opi-vlm/npu-kv-probe/activation-audit-fp16'
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=20)
files=[(scripts/name,name) for name in ('activation_probe.c','board_probe.c')]
for r in records:
    files.append((folder/f'fp16/l{r["layer"]}/export_nbg_unify/network_binary.nb',r['name']+'.nb'))
    files.append((folder/(r['name']+'-input.f32'),r['name']+'-input.f32'))
hashes=[]
for path,name in files:
    subprocess.run(scp+[str(path),'root@192.168.88.29:'+remote+'/'+name],check=True,timeout=120)
    hashes.append(hashlib.file_digest(path.open('rb'),'sha256').hexdigest()+'  '+name)
subprocess.run(ssh+['cd '+remote+' && sha256sum -c -'],input='\n'.join(hashes)+'\n',text=True,check=True,timeout=20)
(folder/'deployed-sha256.txt').write_text('\n'.join(hashes)+'\n')
commands=['cd '+remote,'cc -O2 -Wall -Wextra -Werror activation_probe.c -lNBGlinker -lVIPhal -lm -o activation_probe']
for r in records:
    name=r['name']
    commands.append(f'timeout 60 ./activation_probe {name}.nb {r["count"]} {r["cases"]} {name}-input.f32 {name}-output.f32')
command='flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(' && '.join(commands))
with (folder/'run.log').open('w') as log:
    result=subprocess.run(ssh+[command],stdout=log,stderr=subprocess.STDOUT,timeout=2100)
(folder/'exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
result.check_returncode()
for r in records:
    name=r['name']+'-output.f32'
    subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+name,str(folder/name)],check=True,timeout=30)
