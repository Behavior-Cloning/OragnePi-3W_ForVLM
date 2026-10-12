"""Deploy a resized decoder probe and compare every output on the board."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--tag',required=True)
p.add_argument('--max-l2',type=float,default=.05);p.add_argument('--min-cosine',type=float,default=.99)
a=p.parse_args();assert a.tag.replace('-','').isalnum()
scripts=Path(__file__).resolve().parent;remote='/opt/opi-vlm/npu-kv-probe/'+a.tag
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','root@192.168.88.29'];scp=prefix+['scp','-o','ForwardX11=no']
subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=20)
files=[(scripts/f,f) for f in ('tensor_probe.c','full_model.c','board_probe.c','greedy_i16.h','batch_prefill.h')]
files += [(a.folder/'export_nbg_unify/network_binary.nb','model.nb'),(a.folder/'probe-input.f32','input.f32')]
for source,name in files:subprocess.run(scp+[str(source),'root@192.168.88.29:'+remote+'/'+name],check=True,timeout=120)
command='cd '+remote+' && cc -O2 -Wall -Wextra -Werror tensor_probe.c -lNBGlinker -lVIPhal -lm -o probe && ./probe model.nb input.f32 output.f32 5 timing.json'
with (a.folder/'board-probe.log').open('w') as log:
    subprocess.run(ssh+['flock -w 120 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
for file in ('output.f32','timing.json'):subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+file,str(a.folder/file)],check=True,timeout=60)
expected=np.fromfile(a.folder/'probe-reference.f32',np.float32);actual=np.fromfile(a.folder/'output.f32',np.float32)
assert actual.shape==expected.shape and np.isfinite(actual).all()
fixture=json.loads((a.folder/'probe-fixture.json').read_text());rows=[];offset=0
for name,count in fixture['outputs']:
    x,y=actual[offset:offset+count],expected[offset:offset+count];offset+=count
    rows.append({'tensor':name,'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)),
                 'cosine':float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20))})
passed=all(r['relative_l2']<=a.max_l2 and r['cosine']>=a.min_cosine for r in rows)
report={'fixture':fixture,'errors':rows,'timing':json.loads((a.folder/'timing.json').read_text()),
        'gate':{'max_relative_l2':a.max_l2,'min_cosine':a.min_cosine,'passed':passed}}
(a.folder/'board-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 3)
