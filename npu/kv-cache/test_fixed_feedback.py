"""On-board runner guards and deliberate EOS override with fixed feedback."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parents[2]
remote='/opt/opi-vlm/npu-kv-probe/full-smol500-c128-w8-fused2-gqa'
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
code=r'''
import json,os,pathlib,struct,subprocess,tempfile
root=pathlib.Path(REMOTE);os.chdir(root)
reference=json.loads((root/'board-companion/tokens.json').read_text())
base=['./board-mlp-fixed-full_model','.', '32','128','49280','embeddings.f32','board-mlp-fixed-prompt.i32']
records=[]
with tempfile.TemporaryDirectory(prefix='feedback-guards-',dir=root) as name:
    folder=pathlib.Path(name)
    def run(label,data,steps,eos,expected):
        env=os.environ.copy()
        for k in ('OPI_PERFORMANCE','OPI_FULL_TRACE','OPI_FEEDBACK_FILE'):env.pop(k,None)
        env.update(OPI_DECODER_GROUP='16',OPI_FAST_GREEDY='1')
        if data is not None:
            path=folder/(label+'.i32');path.write_bytes(data);env['OPI_FEEDBACK_FILE']=str(path)
        output=folder/(label+'.json');logits=folder/(label+'.f32')
        p=subprocess.run(base+[str(steps),str(output),str(logits),str(eos)],env=env,capture_output=True,timeout=60)
        assert p.returncode==expected,(label,p.returncode,p.stderr.decode()[-500:])
        row={'case':label,'exit':p.returncode,'stderr_tail':p.stderr.decode()[-120:]}
        if expected==0:
            r=json.loads(output.read_text());assert r['driver_after_release_bytes']==0
            row['result']=r
            count=r['generated_tokens'];assert logits.read_bytes()==(root/'board-companion/logits.f32').read_bytes()[:count*49280*4]
        records.append(row)
        return row
    run('short',struct.pack('<i',339),2,49279,2)
    run('long',struct.pack('<iii',339,1,2),2,49279,2)
    run('negative',struct.pack('<ii',-1,1),2,49279,2)
    run('past-vocabulary',struct.pack('<ii',49280,1),2,49279,2)
    run('over-capacity',struct.pack('<128i',*([1]*128)),128,49279,2)
    # Deliberately designate the known first prediction as EOS. Free decoding
    # stops at it, while fixed-work decoding must still execute the second step.
    first=reference['tokens'][0]
    natural=run('natural-eos',None,2,first,0)
    fixed=run('fixed-eos',struct.pack('<ii',*reference['tokens'][:2]),2,first,0)
    assert natural['result']['generated_tokens']==1 and not natural['result']['fixed_feedback']
    assert fixed['result']['generated_tokens']==2 and fixed['result']['fixed_feedback']
print(json.dumps({'cases':records,'passed':True}))
'''.replace('REMOTE',repr(remote),1)
p=subprocess.run(ssh+['flock -w 60 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code)],
    capture_output=True,text=True,timeout=240,check=True)
report=json.loads(p.stdout)
path=root/'npu/results/fixed-feedback-guards.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print('PASS: malformed feedback/capacity rejected; EOS override and complete logits verified.')
