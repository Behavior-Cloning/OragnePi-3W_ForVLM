#!/usr/bin/env python3
"""Bounded real NPU compiler attempt; does not access the board."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time,uuid,shutil
p=argparse.ArgumentParser();p.add_argument('--variant',default='fp16');a=p.parse_args()
r=Path(__file__).resolve().parent;b=Path('/opt/opi-lab/paligemma2-npu-compat-20261023');o=b/a.variant
assert shutil.disk_usage('/mnt/f').free>8*1024**3
name='pali-compat-'+uuid.uuid4().hex[:12]
cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v','/opt/opi-lab:/work','-w','/work/paligemma2-npu-compat-20261023/'+a.variant,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.monotonic();status={'variant':a.variant,'command':cmd,'board_executed':False,'whole_model':False}
with (o/'export.log').open('w') as f:
 try:status['returncode']=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','-t','2',name],stdout=f,stderr=subprocess.STDOUT,timeout=15);status['returncode']=124
status['seconds']=time.monotonic()-t
nbg=o/'export_nbg_unify/network_binary.nb'
if nbg.exists():status.update(nbg=str(nbg),bytes=nbg.stat().st_size,sha256=hashlib.file_digest(nbg.open('rb'),'sha256').hexdigest())
status['free_F_bytes']=shutil.disk_usage('/mnt/f').free
(o/'compile-result.json').write_text(json.dumps(status,indent=2));(r/('compile-'+a.variant+'.json')).write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
raise SystemExit(status['returncode'])
