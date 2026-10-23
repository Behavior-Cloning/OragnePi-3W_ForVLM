from pathlib import Path
import numpy as np,yaml,json,subprocess,os,time,sys,copy,shutil
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');name=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'gate'
src=b/('w8scaled-all' if name=='l0' else 'w8cal-all')/name;o=b/('w8hybrid-'+variant)/name;o.mkdir(parents=True,exist_ok=True);g=json.loads((src/'decoder.json').read_text());q=yaml.safe_load((src/'decoder.quantize').read_text());nodes=g['Layers'];qp=q['quantize_parameters'];cal=yaml.safe_load((b/('w8mm-cal' if name=='l0' else 'w8retry-i16')/name/'decoder.quantize').read_text())['quantize_parameters'];changes=[]
for i,(n,v) in enumerate(list(nodes.items())):
 if v['op']!='matmul':continue
 if v['name']=='MatMul_1_output_0':
  v['op']='multiply';v['parameters']={'axis':1,'bias':False};changes.append({'name':n,'change':'FP16 broadcast outer product'});continue
 old=list(v['inputs'])
 for j,ref in enumerate(old):
  qi=copy.deepcopy(cal[ref]);assert qi['qtype']=='i16';qi['fl']-=2;conv='mm_input_convert_'+str(30000+i*2+j)
  nodes[conv]={'name':conv,'op':'dtype_converter','parameters':{},'inputs':[ref],'outputs':['out0']};qp['@'+conv+':out0']=qi;v['inputs'][j]='@'+conv+':out0'
 if variant=='i16':
  qo=copy.deepcopy(cal['@'+n+':out0']);qo['fl']-=2;qp['@'+n+':out0']=qo
 changes.append({'name':n,'old':old,'inputs':v['inputs'],'output':qp['@'+n+':out0']})
assert len(changes)==(3 if name=="l0" else 2)
if not (o/'decoder.data').exists():os.link(src/'decoder.data',o/'decoder.data')
(o/'decoder.json').write_text(json.dumps(g,indent=2));(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'changes.json').write_text(json.dumps(changes,indent=2));shutil.copyfile(src/'tensor-inputmeta.yml',o/'tensor-inputmeta.yml')
container='minicpm46-mmint-'+name+'-'+variant
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'variant':variant,'exit':code,'seconds':time.time()-t,'changes':changes,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
