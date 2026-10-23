from pathlib import Path
import numpy as np,yaml,json,subprocess,shutil,os,time,sys,copy
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023')
name=sys.argv[1];mode=sys.argv[2] if len(sys.argv)>2 else 'perchannel';src=b/('w8retry-'+mode)/name;o=b/('w8convert-'+mode)/name;o.mkdir(parents=True,exist_ok=True)
for f in ['decoder.data','tensor-inputmeta.yml']:
 if not (o/f).exists():os.link(src/f,o/f)
g=json.loads((src/'decoder.json').read_text());q=yaml.safe_load((src/'decoder.quantize').read_text());qp=q['quantize_parameters'];changes=[]
for i,(n,v) in enumerate(list(g['Layers'].items())):
 if v['op']!='fullconnect':continue
 before=v['inputs'][0];conv='fc_input_convert_'+str(20000+i)
 assert g['Layers'][before[1:].split(':')[0]]['op']=='reshape'
 calibrated=copy.deepcopy(qp[before]);qp[before]={'qtype':'float16','quantizer':'float16'}
 g['Layers'][conv]={'name':conv,'op':'dtype_converter','parameters':{},'inputs':[before],'outputs':['out0']}
 qp['@'+conv+':out0']=calibrated;v['inputs'][0]='@'+conv+':out0';changes.append({'fc':n,'converter':conv,'quant':calibrated})
(o/'decoder.json').write_text(json.dumps(g,indent=2));(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'changes.json').write_text(json.dumps(changes,indent=2))
container='minicpm46-convert-'+name+'-'+mode
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'mode':mode,'exit':code,'seconds':time.time()-t,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
