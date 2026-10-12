"""Check poisoned unused KV and reject over-capacity runs on the real NPU."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np

root=Path(__file__).resolve().parents[2]
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
results=[]
for dtype in ('int16','fp16'):
    tag=f'trained-smol-c128-{dtype}'
    remote='/opt/opi-vlm/npu-kv-probe/'+tag
    local=root/'npu/results/trained-smol-c128'/tag
    run=subprocess.run(ssh+[f'cd {remote} && OPI_KV_POISON=1 timeout 30 ./board_probe layer0.nb layer1.nb 128 128 hidden.f32 reference.f32 poison.f32'],
                       capture_output=True,text=True,timeout=40)
    (local/'poison.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stderr
    subprocess.run(scp+['root@192.168.88.29:'+remote+'/poison.f32',str(local/'poison.f32')],check=True,timeout=20)
    normal=np.fromfile(local/'output.f32',np.float32);poison=np.fromfile(local/'poison.f32',np.float32)
    np.testing.assert_array_equal(normal,poison)
    overflow=subprocess.run(ssh+[f'cd {remote} && ./board_probe layer0.nb layer1.nb 128 129 hidden.f32 reference.f32 rejected.f32'],
                            capture_output=True,text=True,timeout=20)
    assert overflow.returncode==2,overflow.stdout+overflow.stderr
    results.append({'tag':tag,'poisoned_unused_cache':'PASS; byte-identical output',
                    'max_poison_difference':float(np.max(np.abs(normal-poison))),
                    'capacity_overflow_exit_code':overflow.returncode,
                    'measurement':next(json.loads(line) for line in run.stdout.splitlines() if line.startswith('{'))})
out=Path(__file__).resolve().parent/'evidence-20261012/board-guards.json'
out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
