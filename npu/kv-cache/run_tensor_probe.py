"""Deploy a resized decoder probe and compare every output on the board."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--tag',required=True)
p.add_argument('--max-l2',type=float,default=.05);p.add_argument('--min-cosine',type=float,default=.99)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--npd',action='store_true',help='enable SDK operator profiling; timing is diagnostic only')
p.add_argument('--result-dir',type=Path,help='retain each new measurement without overwriting fixture evidence')
p.add_argument('--cpu-affinity',help='optional taskset CPU list')
p.add_argument('--resident-copies',type=int,default=0,help='diagnostic identical prepared graphs interleaved before primary execution')
p.add_argument('--alternate',type=Path,help='another validated fixture with identical graph: alternate inputs and verify repeatability for each phase')
a=p.parse_args();assert a.tag.replace('-','').isalnum()
if a.cpu_affinity:assert 0<len(a.cpu_affinity)<=24 and all(c in '01234567,-' for c in a.cpu_affinity)
result_dir=a.result_dir or a.folder
if a.result_dir:result_dir.mkdir(parents=True,exist_ok=False)
assert 1<=a.repeats<=100
assert 0<=a.resident_copies<=16 and not (a.npd and a.resident_copies)
from board_host_lock import lock_board_from_host
lock_board_from_host()
scripts=Path(__file__).resolve().parent;remote='/opt/opi-vlm/npu-kv-probe/'+a.tag
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','root@192.168.88.29'];scp=prefix+['scp','-o','ForwardX11=no']
subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=20)
files=[(scripts/f,f) for f in ('tensor_probe.c','full_model.c','board_probe.c','greedy_i16.h','batch_prefill.h')]
files += [(a.folder/'export_nbg_unify/network_binary.nb','model.nb'),(a.folder/'probe-input.f32','input.f32')]
if a.alternate:
    assert a.repeats>=4 and a.repeats%2==0
    assert (a.folder/'export_nbg_unify/network_binary.nb').read_bytes()==(a.alternate/'export_nbg_unify/network_binary.nb').read_bytes()
    assert (a.folder/'probe-input.f32').stat().st_size==(a.alternate/'probe-input.f32').stat().st_size
    files.append((a.alternate/'probe-input.f32','alternate.f32'))
for source,name in files:subprocess.run(scp+[str(source),'root@192.168.88.29:'+remote+'/'+name],check=True,timeout=120)
command='cd '+remote+' && cc -O2 -Wall -Wextra -Werror tensor_probe.c -lNBGlinker -lVIPhal -lm -o probe && ./probe model.nb input.f32 output.f32 '+str(a.repeats)+' timing.json'
if a.alternate:command=command.replace('&& ./probe','&& OPI_PROBE_ALTERNATE=alternate.f32 ./probe')
if a.npd:command=command.replace('./probe model.nb','env OPI_PROBE_NPD=1 ./probe model.nb')
if a.resident_copies:command=command.replace('./probe model.nb','env OPI_PROBE_RESIDENT_COPIES='+str(a.resident_copies)+' ./probe model.nb')
if a.cpu_affinity:command=command.replace('./probe model.nb','taskset -c '+a.cpu_affinity+' ./probe model.nb')
with (result_dir/'board-probe.log').open('w') as log:
    subprocess.run(ssh+['flock -w 120 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
for file in ('output.f32','timing.json'):subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+file,str(result_dir/file)],check=True,timeout=60)
if a.npd:
    names=subprocess.check_output(ssh+['find '+remote+' -maxdepth 1 -name "*.npd" -type f'],text=True,timeout=20).splitlines()
    assert names,'SDK emitted no NPD files; inspect board-probe.log'
    for name in names:subprocess.run(scp+['root@192.168.88.29:'+name,str(result_dir/Path(name).name)],check=True,timeout=60)
expected=np.fromfile((a.alternate or a.folder)/'probe-reference.f32',np.float32);actual=np.fromfile(result_dir/'output.f32',np.float32)
assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
fixture=json.loads((a.folder/'probe-fixture.json').read_text());rows=[];offset=0
for name,count in fixture['outputs']:
    x,y=actual[offset:offset+count],expected[offset:offset+count];offset+=count
    rows.append({'tensor':name,'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)),
                 'cosine':float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20))})
assert offset==expected.size and rows, 'fixture must cover every output element'
timing=json.loads((result_dir/'timing.json').read_text())
passed=all(r['relative_l2']<=a.max_l2 and r['cosine']>=a.min_cosine for r in rows) and timing['differing_repeats']==0 and timing.get('resident_mismatched_elements',0)==0 and timing['released_bytes']==0
report={'fixture':fixture,'folder':str(a.folder),'cpu_affinity':a.cpu_affinity,
        'binary_sha256':hashlib.sha256((a.folder/'export_nbg_unify/network_binary.nb').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((a.folder/'probe-input.f32').read_bytes()).hexdigest(),
        'alternate_fixture':json.loads((a.alternate/'probe-fixture.json').read_text()) if a.alternate else None,'errors':rows,'timing':timing,
        'gate':{'max_relative_l2':a.max_l2,'min_cosine':a.min_cosine,'passed':passed}}
(result_dir/'board-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 3)
