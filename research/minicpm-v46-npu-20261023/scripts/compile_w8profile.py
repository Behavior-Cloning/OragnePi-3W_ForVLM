from pathlib import Path
import numpy as np,yaml,json,subprocess,os,time,sys,copy,shutil
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');name=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'gate'
src=b/'w8scaled-all'/name;o=b/'w8profile'/name;o.mkdir(parents=True,exist_ok=True);changes=[]
for f in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml']:
 if not (o/f).exists():os.link(src/f,o/f)
container='minicpm46-profile-'+name+'-'+variant
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','2','--memory','5g','-e','VIV_VX_DUMP_NODE_TRACE_DB=1','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'variant':variant,'exit':code,'seconds':time.time()-t,'changes':changes,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
