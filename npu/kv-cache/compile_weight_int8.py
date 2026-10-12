"""Bounded weight-only INT8 export; successful export alone is not acceleration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import numpy as np
import yaml

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=180)
args=p.parse_args();root=Path(__file__).resolve().parents[2]/'npu/results'
base=args.base.resolve();out=args.output.resolve();relative=out.relative_to(root)
out.mkdir(parents=True,exist_ok=True)
for name in ('decoder.json','decoder.data','tensor-inputmeta.yml'):
    shutil.copyfile(base/name,out/name)
quant=yaml.safe_load((base/'decoder.quantize').read_text())
# This is our own compiler-generated numpy dictionary, not an external pickle.
data=np.load(base/'decoder.data',allow_pickle=True).item();changed=[]
for name,parameters in data.items():
    key=f'@{name}:weight'
    if name.startswith('fullconnect') and key in quant['quantize_parameters']:
        weight=parameters['weight']
        scales=np.maximum(np.max(np.abs(weight),axis=0)/127,np.finfo(np.float32).tiny)
        quant['quantize_parameters'][key]={'qtype':'i8','quantizer':'perchannel_symmetric_affine',
            'rounding':'rtne','scale':scales.tolist(),'channel_dim':-1}
        changed.append({'name':name,'shape':list(weight.shape)})
assert len(changed)==7, 'this experiment expects one decoder with seven linear projections'
(out/'decoder.quantize').write_text(yaml.safe_dump(quant,sort_keys=False))
report={'scope':'one decoder; INT8 per-channel linear weights; other tensor precisions preserved from base',
        'base':str(base.relative_to(root)),'weights':changed,
        'source_quantize_sha256':hashlib.file_digest((base/'decoder.quantize').open('rb'),'sha256').hexdigest()}
container='opi-w8-test-'+uuid.uuid4().hex[:12]
command=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g',
    '-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4',
    '-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B',
    '-v',str(root)+':/work','-w','/work/'+str(relative),'ubuntu-npu:v2.0.10.2',
    'python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib',
    '--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize',
    '--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64',
    '--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B',
    '--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
start=time.monotonic()
with (out/'export.log').open('w') as log:
    try:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        report['exit_code']=proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','-t','2',container],stdout=log,stderr=subprocess.STDOUT,timeout=15)
        report['exit_code']=124
report['seconds']=time.monotonic()-start
report['binaries']=[{'bytes':f.stat().st_size,'path':str(f.relative_to(out))} for f in out.rglob('network_binary.nb')]
(out/'conversion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
raise SystemExit(report['exit_code'])
