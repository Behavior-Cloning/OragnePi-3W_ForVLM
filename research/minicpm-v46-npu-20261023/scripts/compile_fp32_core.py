from pathlib import Path
import subprocess,yaml,shutil,json
b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');r=Path(__file__).resolve().parents[1];name='core-fp32';o=b/name;o.mkdir(exist_ok=True);src=b/'matmul-b1-fp16'
for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:shutil.copyfile(src/n,o/n)
q=yaml.safe_load((src/'decoder.quantize').read_text())
for key in q['quantize_parameters']:q['quantize_parameters'][key]={'qtype':'float32','quantizer':'float32'}
(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
cmd=['docker','run','--rm','--name','minicpm46-fp32-core','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2','minicpm46-fp32-core']);code=124
(r/'results/fp32-core-compile.json').write_text(json.dumps({'exit':code,'scope':'all core graph tensors/constants requested as FP32'}));print(code,flush=True)
