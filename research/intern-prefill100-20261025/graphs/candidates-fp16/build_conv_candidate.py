#!/usr/bin/env python3
from pathlib import Path
import copy,hashlib,json,subprocess,sys,time,traceback,uuid,re,shutil
import numpy as np,torch,yaml
R=Path(__file__).resolve().parent;G=R.parent;TASK=G.parent;REPO=TASK.parents[1];SRC=G/'cases/c512-m16-l2';OUT=R/'conv1x1-m16-l2';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
refsrc=(REPO/'npu/kv-cache/graph_reference.py').read_text();needle="        elif op=='reshape':y=x[0].reshape(p['shape'])";assert needle in refsrc
refsrc=refsrc.replace(needle,"        elif op=='convolution':\n            w=torch.from_numpy(data[key]['weight']).permute(3,2,0,1).contiguous()\n            y=F.conv2d(x[0].permute(0,3,1,2).contiguous(),w,torch.from_numpy(data[key]['bias'])).permute(0,2,3,1).contiguous()\n"+needle)
(R/'graph_reference_conv.py').write_text(refsrc);space={};exec(compile(refsrc,str(R/'graph_reference_conv.py'),'exec'),space);evaluate=space['evaluate']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (R/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'state':'preparing','start':time.time(),'baseline':str(SRC),'candidate':str(OUT)};save()
try:
 original=json.loads((SRC/'decoder.json').read_text());g=copy.deepcopy(original);data=np.load(SRC/'decoder.data',allow_pickle=True).item();d=copy.deepcopy(data);q=(SRC/'decoder.quantize').read_bytes();schema=json.loads((R/'schema.json').read_text());params=next(x['parameters'] for x in schema['Layers'].values() if x['op']=='convolution');changes=[]
 for b in [0,1]:
  shape=f'b{b}_reshape_115';key=f'b{b}_fullconnect_116';assert g['Layers'][shape]['parameters']['shape']==[-1,4864]
  g['Layers'][shape]['parameters']['shape']=[1,1,16,4864];g['Layers'][key]['op']='convolution';g['Layers'][key]['parameters']={**params,'weights':896}
  assert d[key]['weight'].shape==(4864,896);d[key]['weight']=d[key]['weight'].reshape(1,1,4864,896)
  assert d[key]['weight'].tobytes()==data[key]['weight'].tobytes()
  changes.append({'key':key,'old_op':'fullconnect','new_op':'convolution','input_shape':[1,1,16,4864],'weight_shape':[1,1,4864,896],'output_shape':[1,1,16,896],'same_values':True})
 for name,vals in data.items():
  for field,v in vals.items():assert d[name][field].tobytes()==v.tobytes(),(name,field)
 (OUT/'decoder.json').write_text(json.dumps(g,indent=2)+'\n')
 with (OUT/'decoder.data').open('wb') as f:np.save(f,d,allow_pickle=True)
 (OUT/'decoder.quantize').write_bytes(q);shutil.copy2(SRC/'tensor-inputmeta.yml',OUT/'tensor-inputmeta.yml')
 meta=json.loads((TASK/'fixtures-canonical/c512-m16-l2/probe-fixture.json').read_text());raw=np.fromfile(TASK/'fixtures-canonical/c512-m16-l2/probe-input.f32',np.float32);feeds={};off=0
 for k,n in meta['inputs']:
  shape=next(x['shape'] for x in meta['input_contract'] if x['name']==k);feeds[k]=torch.from_numpy(raw[off:off+n].reshape(shape).copy());off+=n
 torch.set_num_threads(2);errors={}
 with torch.inference_mode():
  for mode in ['source-f32','down-weight-f16-reexpanded']:
   a=copy.deepcopy(data);b=copy.deepcopy(d)
   if mode=='down-weight-f16-reexpanded':
    for block in [0,1]:
     key=f'b{block}_fullconnect_116'
     a[key]['weight']=a[key]['weight'].astype(np.float16).astype(np.float32);b[key]['weight']=b[key]['weight'].astype(np.float16).astype(np.float32)
     a[key]['bias']=a[key]['bias'].astype(np.float16).astype(np.float32);b[key]['bias']=b[key]['bias'].astype(np.float16).astype(np.float32)
   old=evaluate(original,a,feeds);new=evaluate(g,b,feeds);rows={}
   for k,v in new.items():
    rel=float(torch.linalg.vector_norm(v-old[k])/torch.linalg.vector_norm(old[k]));mx=float((v-old[k]).abs().max());torch.testing.assert_close(v,old[k],atol=1e-4,rtol=5e-4);assert rel<1e-5
    rows[k]={'relative_l2':rel,'maxabs':mx,'finite':bool(torch.isfinite(v).all())}
   errors[mode]=rows
 state.update(state='fp32-equivalence-passed',changes=changes,graph_equivalence=errors,quantizer_byte_equal=sha(OUT/'decoder.quantize')==sha(SRC/'decoder.quantize'),source_hashes={n:sha(SRC/n) for n in ['decoder.json','decoder.data','decoder.quantize','export/decoder.export.data','export_nbg_unify/network_binary.nb']},candidate_graph_sha256=sha(OUT/'decoder.json'),candidate_data_sha256=sha(OUT/'decoder.data'),source_reference_sha256=sha(REPO/'npu/kv-cache/graph_reference.py'),local_reference_sha256=sha(R/'graph_reference_conv.py'));save()
 # Serialize compilation after the trace exports to respect the F-drive/resource budget.
 deadline=time.time()+300
 while (G/'trace-export/status.json').exists() and json.loads((G/'trace-export/status.json').read_text())['state']=='running':
  assert time.time()<deadline,'trace export failed to finish';time.sleep(2)
 assert json.loads((G/'trace-export/status.json').read_text())['state']=='complete'
 cmd=json.loads((SRC/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(SRC),str(OUT)) for x in cmd];cmd[cmd.index('--name')+1]='opi-conv1x1-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1'];state.update(state='compiling',compile_command=cmd);save();start=time.monotonic()
 with (OUT/'export.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 assert p.returncode==0,('compiler rc',p.returncode)
 conv={'first_layer':2,'fusion_width':2,'capacity':512,'query_tokens':16,'scope':'FP16 down FCL->NHWC 1x1 convolution; original quantizer unchanged; kernel weight reshape only','compile_command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-start,**verify_contract(OUT,2)};(OUT/'conversion.json').write_text(json.dumps(conv,indent=2)+'\n')
 def weights(folder,key):
  code=(folder/'export/vnn_decoder.c').read_text();block=code.split('/* @'+key+':weight */',1)[1].split('/*',1)[0];match=re.search(r'VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)',block);assert match,(folder,key);dtype,offset,count=match.groups();raw=(folder/'export/decoder.export.data').read_bytes();return dtype,raw[int(offset):int(offset)+int(count)]
 records=[]
 for key,node in original['Layers'].items():
  if node['op']!='fullconnect':continue
  ta,wa=weights(SRC,key);tb,wb=weights(OUT,key);assert ta==tb and wa==wb,(key,ta,tb,'weight payload mismatch');records.append({'key':key,'dtype':ta,'sha256':hashlib.sha256(wb).hexdigest(),'bytes':len(wb),'source_exported_weight_exact':True})
 oldmeta=json.loads((SRC/'export_nbg_unify/nbg_meta.json').read_text());newmeta=json.loads((OUT/'export_nbg_unify/nbg_meta.json').read_text());assert oldmeta==newmeta,'external port type or shape changed'
 nb=OUT/'export_nbg_unify/network_binary.nb';trace=[str(p) for p in OUT.rglob('*trace*') if p.is_file()]
 state.update(state='complete',end=time.time(),compile_seconds=conv['seconds'],nb=str(nb),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size,all_14_exported_fc_weight_dtype_and_bytes_same=records,all_external_ports_exact=oldmeta==newmeta,trace_files=trace,board_numeric_status='pending; no board operations performed');save();print(json.dumps({k:v for k,v in state.items() if k in ['state','nb','nb_sha256','nb_bytes','compile_seconds','trace_files']}),flush=True)
except BaseException:state.update(state='failed',end=time.time(),error=traceback.format_exc());save();raise
