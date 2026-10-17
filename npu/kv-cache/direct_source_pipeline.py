"""Fresh source-weight export, numerical checks, and serial calibration/build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
p.add_argument('--resume',action='store_true');a=p.parse_args()
a.model=a.model.resolve();a.work=a.work.resolve()
scripts=Path(__file__).resolve().parent;source=a.work/'export';measure=a.work/'measurements'
a.work.mkdir(parents=True,exist_ok=a.resume);measure.mkdir(exist_ok=True)

def run(name,command):
    target=measure/name
    if target.exists():
        state=target/'summary.json'
        if a.resume and state.exists() and json.loads(state.read_text())['exit_code']==0:return
        raise RuntimeError(f'incomplete prior stage: {target}; inspect before retry')
    (a.work/'active-stage.txt').write_text(name)
    print('START',name,flush=True)
    subprocess.run([sys.executable,str(scripts/'measure_wsl_command.py'),'--output',str(target),'--',*map(str,command)],check=True)

run('export',[sys.executable,scripts/'trained_probe.py','--model',a.model,'--output',source,'--layers','0','--capacity','128','--tokens','64','--stable-norm'])
run('standard-gguf-oracle',[sys.executable,scripts/'verify_gguf_export.py','--model',a.model,'--source',source])
run('shard-head',[sys.executable,scripts/'export_head_shards.py','--source',source])
m=json.loads((source/'trained-manifest.json').read_text());records=[]
for i in range(len(m['layers'])+len(m['head_shards'])):
    base=source/'int16'/f'l{i}';out=source/'w8-balanced'/f'l{i}'
    run(f'calibrate-{i:02d}',[sys.executable,scripts/'compile_probe.py','--results-root',a.work,'--source',source,'--output',base,'--layer',i,'--dtype','int16'])
    cmd=[sys.executable,scripts/'compile_weight_int8.py','--results-root',a.work,'--base',base,'--output',out]
    if i<len(m['layers']):
        cmd+=['--value-dfp','--float-residual','--float-sensitive','--activation-headroom-bits','1']
        if i==2:cmd+=['--float-mlp']
    else:cmd+=['--expected-linears','1','--float-head-norm','--weight-mode','dfp','--dfp-method','mse']
    run(f'balanced-{i:02d}',cmd)
    binary=out/'export_nbg_unify/network_binary.nb'
    with binary.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    records.append({'layer':i,'exit_code':0,'path':str(binary),'sha256':digest,'bytes':binary.stat().st_size})
    (source/'w8-balanced/stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
(a.work/'active-stage.txt').write_text('BUILD_COMPLETE_BOARD_VALIDATION_PENDING')
(a.work/'build-manifest.json').write_text(json.dumps({'source_identity':m['source_identity'],'status':'COMPILED_NOT_BOARD_VALIDATED','fresh_calibration':True,'calibration_positions':[0,1,2,4,8,32,63],'records':records},indent=2)+'\n')
print('ALL_28_FRESHLY_CALIBRATED_AND_BUILT',flush=True)
