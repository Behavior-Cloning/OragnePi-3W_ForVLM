"""Board integration guards for batched tensor dimensions and timing boundaries."""
import json
import os
from pathlib import Path
import shlex
import subprocess

code=r'''
import json,os,pathlib,subprocess,tempfile
root=pathlib.Path('/opt/opi-vlm/npu-kv-probe/full-intern1b-c256-batch20-vslice');os.chdir(root)
base=['./board-history104-fixed-batch-r9-full_model','.','24','256','151674','embeddings.f32','board-history104-fixed-batch-r9-prompt.i32']
cases=[('negative-history',{'OPI_CACHED_PREFIX':'-1'},2),
       ('whole-prompt-as-history',{'OPI_CACHED_PREFIX':'104'},2),
       ('invalid-batch-one',{'OPI_PREFILL_TOKENS':'1'},2),
       ('batch-too-large',{'OPI_PREFILL_TOKENS':'257'},2),
       ('wrong-compiled-batch',{'OPI_PREFILL_TOKENS':'19'},2),
       ('invalid-scalar-prefix',{'OPI_PREFILL_SCALAR_PREFIX':'105'},2)]
records=[]
with tempfile.TemporaryDirectory(prefix='prefill-guards-',dir=root) as folder:
    for label,override,expected in cases:
        env=os.environ.copy()
        for key in list(env):
            if key.startswith('OPI_'):env.pop(key)
        env.update(OPI_FAST_GREEDY='1',OPI_PREFILL_TOKENS='20',OPI_DECODER_GROUP='12')
        env.update(override)
        p=subprocess.run(base+['2',folder+'/tokens.json',folder+'/logits.f32','151645'],env=env,capture_output=True,timeout=90)
        assert p.returncode==expected,(label,p.returncode,p.stderr.decode()[-500:])
        records.append({'case':label,'exit_code':p.returncode,'stderr_tail':p.stderr.decode()[-180:]})
print(json.dumps({'passed':True,'cases':records}))
'''
if __name__=='__main__':
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    r=subprocess.run(prefix+['ssh','-o','ForwardX11=no','root@192.168.88.29',
        'flock -w 1800 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code)],
        check=True,capture_output=True,text=True,timeout=2400)
    result=json.loads(r.stdout)
    (Path(__file__).resolve().parents[2]/'npu/results/prefill-guards.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
