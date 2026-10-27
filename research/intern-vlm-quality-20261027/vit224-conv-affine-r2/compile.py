from pathlib import Path
import json,copy,hashlib,shutil,subprocess,time,fcntl,ast
import numpy as np
import torch,yaml
P=Path(__file__).resolve().parent;R=P.parents[2];S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vitfull-20261027-r1');base=S/'w8-native224-r1-l00';out=S/'conv-affine-r2-l00';out.mkdir(exist_ok=False)
state={'status':'preparing','scope':'single ViT block00, six linears represented as 1x1 convolution; same Transformer graph semantics, equivalent integer grid, FC input quantizer changed from DFP to zero-point0 affine; no board result yet'}
def save():(P/'compile-status.json').write_text(json.dumps(state,indent=2)+'\n')
save();g=json.loads((base/'decoder.json').read_text());original=copy.deepcopy(g);data=np.load(base/'decoder.data',allow_pickle=True).item();d=copy.deepcopy(data)
schema=json.loads((R/'research/intern-prefill100-20261025/graphs/candidates-fp16/schema.json').read_text());params=next(n['parameters'] for n in schema['Layers'].values() if n['op']=='convolution');changes=[];torch.set_num_threads(2);torch.manual_seed(20261027)
for k,v in g['Layers'].items():
 if v['op']!='fullconnect':continue
 w=data[k]['weight'];ni,no=w.shape;pre=v['inputs'][0][1:].split(':')[0];node=g['Layers'][pre];assert node['op']=='reshape' and node['parameters']['shape']==[-1,ni]
 users=[n for n,z in original['Layers'].items() if v['inputs'][0] in z['inputs']];assert users==[k]
 node['parameters']['shape']=[1,1,257,ni];v['op']='convolution';v['parameters']={**params,'weights':no};d[k]['weight']=w.reshape(1,1,ni,no)
 x=torch.randn(7,ni);weight=torch.tensor(w.copy());bias=torch.tensor(data[k]['bias'].copy())
 y=x@weight+bias;z=torch.nn.functional.conv2d(x.T.reshape(1,ni,1,7),weight.T.reshape(no,ni,1,1),bias).reshape(no,7).T
 rel=float(torch.linalg.norm(y-z)/torch.linalg.norm(y));assert rel<1e-5;changes.append({'node':k,'in':ni,'out':no,'FP32_equivalence_rel_l2':rel})
assert len(changes)==6
for k,vals in data.items():
 for field,v in vals.items():assert v.tobytes()==d[k][field].tobytes(),(k,field)
(out/'decoder.json').write_text(json.dumps(g,indent=2)+'\n')
with (out/'decoder.data').open('wb') as f:np.save(f,d,allow_pickle=True)
q=yaml.safe_load((base/'decoder.quantize').read_text());changed_quant=[]
for k,v in original['Layers'].items():
 if v['op']!='fullconnect':continue
 key=v['inputs'][0];old=q['quantize_parameters'][key].copy();assert old['qtype']=='i16' and old['quantizer']=='dynamic_fixed_point';after=old.copy();fl=after.pop('fl');after.update(quantizer='asymmetric_affine',scale=2.**(-fl),zero_point=0);q['quantize_parameters'][key]=after;codes=np.arange(-32768,32768,dtype=np.float32);assert np.array_equal(codes*np.float32(2.**(-fl)),(codes-0)*np.float32(after['scale']));changed_quant.append({'key':key,'before':old,'after':after})
(out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));shutil.copy2(base/'tensor-inputmeta.yml',out/'tensor-inputmeta.yml');state['quantizer_changes']=changed_quant
state.update(status='compiling',changes=changes,raw_weight_values_exact=True,quantize_sha256=hashlib.sha256((out/'decoder.quantize').read_bytes()).hexdigest());save()
cmd=['docker','run','--rm','--name','opi-vit224-conv-affine-r2','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(S)+':/work','-w','/work/'+out.name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (P/'compile.log').open('w') as log:
  try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240);rc=z.returncode
  except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2','opi-vit224-conv-affine-r2'],stdout=log,stderr=subprocess.STDOUT);rc=124
state['returncode']=rc;nb=out/'export_nbg_unify/network_binary.nb'
if rc==0:
 assert json.loads((base/'export_nbg_unify/nbg_meta.json').read_text())==json.loads((out/'export_nbg_unify/nbg_meta.json').read_text())
 state.update(status='compiled-not-board-tested',path=str(nb),sha256=hashlib.sha256(nb.read_bytes()).hexdigest(),external_metadata_exact=True)
else:state['status']='compile-failed'
save();print(json.dumps(state));raise SystemExit(rc)
