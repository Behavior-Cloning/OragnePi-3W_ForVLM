"""Bounded NBG export for already imported/quantized one-layer graph."""
from pathlib import Path
import argparse,json,subprocess,time,uuid
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--timeout',type=int,default=300);a=p.parse_args();d=a.directory.resolve();rel=d.relative_to('/opt/opi-lab');name='llama-full-'+uuid.uuid4().hex[:12]
cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v','/opt/opi-lab:/work','-w','/work/'+str(rel),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.monotonic()
with (d/'export.log').open('w') as f:
 try:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout).returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','-t','2',name],stdout=f,stderr=subprocess.STDOUT,timeout=15);rc=124
(d/'export-result.json').write_text(json.dumps(dict(command=cmd,returncode=rc,seconds=time.monotonic()-t),indent=2));raise SystemExit(rc)
