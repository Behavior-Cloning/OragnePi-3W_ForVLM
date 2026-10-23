from pathlib import Path
import numpy as np,yaml,json,subprocess,os,time,sys,copy,shutil
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');name=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'gate'
src=b/'w8scaled-all'/name;o=b/'w8balanced'/name;o.mkdir(parents=True,exist_ok=True);g=json.loads((src/'decoder.json').read_text());q=yaml.safe_load((src/'decoder.quantize').read_text());nodes=g['Layers'];qp=q['quantize_parameters'];changes=[]
def fp(ref):qp[ref]={'qtype':'float16','quantizer':'float16'}
def add(n,op,inputs,params=None):
 assert n not in nodes;nodes[n]={'name':n,'op':op,'parameters':params or {},'inputs':inputs,'outputs':['out0']};fp('@'+n+':out0');return '@'+n+':out0'
def qi(ref,fl):qp[ref]={'qtype':'i16','quantizer':'dynamic_fixed_point','rounding':'rtne','fl':fl}
for i,(n,v) in enumerate(list(nodes.items())):
 if v['op']!='matmul':continue
 if v['name']=='MatMul_1_output_0':v['op']='multiply';v['parameters']={'axis':1,'bias':False};changes.append({'node':n,'change':'FP16 broadcast outer product'});continue
 assert v['name'] in ['MatMul_output_0','MatMul_2_output_0'];original=list(v['inputs']);sid=40000+i*10
 # Per-head dynamic normalization of the recurrent-state operand.
 ar=add('state_abs_'+str(sid),'abs',[original[1]])
 sr=add('state_max_'+str(sid+1),'reducemax',[ar],{'axis_list':[2,3],'keep_dims':True})
 sr=add('state_floor_'+str(sid+2),'clipbyvalue',[sr],{'clip_value_min':0.0001,'clip_value_max':float('inf')})
 nr=add('state_norm_'+str(sid+3),'real_div',[original[1],sr])
 bi=add('state_i16_'+str(sid+4),'dtype_converter',[nr]);qi(bi,14)
 ai=add('qk_i16_'+str(sid+5),'dtype_converter',[original[0]]);qi(ai,14 if v['name']=='MatMul_output_0' else 12)
 restore='state_restore_'+str(sid+6);oldref='@'+n+':out0';newref='@'+restore+':out0'
 for k,consumer in list(nodes.items()):
  if k!=n:consumer['inputs']=[newref if x==oldref else x for x in consumer.get('inputs',[])]
 v['inputs']=[ai,bi];add(restore,'multiply',[oldref,sr],{'axis':1,'bias':False});changes.append({'node':n,'old':original,'inputs':v['inputs'],'restore':newref})
assert len(changes)==3
if not (o/'decoder.data').exists():os.link(src/'decoder.data',o/'decoder.data')
(o/'decoder.json').write_text(json.dumps(g,indent=2));(o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'changes.json').write_text(json.dumps(changes,indent=2));shutil.copyfile(src/'tensor-inputmeta.yml',o/'tensor-inputmeta.yml')
container='minicpm46-balanced-'+name+'-'+variant
cmd=['docker','run','--rm','--name',container,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/'+str(o.relative_to(b)),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
t=time.time()
with (o/'export.log').open('w') as f:
 try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=z.returncode
 except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',container]);code=124
row={'name':name,'variant':variant,'exit':code,'seconds':time.time()-t,'changes':changes,'binaries':[{'path':str(p),'bytes':p.stat().st_size} for p in o.rglob('network_binary.nb')]};(o/'conversion.json').write_text(json.dumps(row,indent=2));print(row,flush=True)

raise SystemExit(code)
