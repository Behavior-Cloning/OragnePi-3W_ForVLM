#!/usr/bin/env python3
"""C1280 prefill-only L2 FP16 Conv overlay; no board deployment."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,time,traceback,uuid,re,shutil
import numpy as np,torch
R=Path(__file__).resolve().parent;G=R.parent;TASK=G.parent;REPO=TASK.parents[1]
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
SRC=SOURCE/'w8-fused2-c1280-t16-cap1-productwide/l2';BASE512=G/'cases/c512-m16-l2';CONV512=R/'conv1x1-m16-l2';OUT=R/'conv1x1-c1280-m16-l2';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
sys.path.insert(0,str(R));from graph_reference_conv import evaluate
state={'state':'preparing','start':time.time(),'source':str(SRC),'candidate':str(OUT),'runtime_scope':'only replace prefill-layer2.nb; decode and all other blocks/head unchanged'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():(R/'status-c1280.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 evidence=REPO/'research/intern1b-operator-opt-20261023/results/20261023T064955/events.jsonl';known=json.loads(evidence.open().readline())['graph_sha256'];assert sha(SRC/'export_nbg_unify/network_binary.nb')==known['prefill-layer2.nb']
 for name in ['decoder.data','decoder.quantize','export/decoder.export.data']:assert sha(SRC/name)==sha(BASE512/name),(name,'different baseline constants')
 original=json.loads((SRC/'decoder.json').read_text());g=copy.deepcopy(original);data=np.load(SRC/'decoder.data',allow_pickle=True).item();d=np.load(CONV512/'decoder.data',allow_pickle=True).item()
 params=next(x['parameters'] for x in json.loads((R/'schema.json').read_text())['Layers'].values() if x['op']=='convolution')
 for block in [0,1]:
  key=f'b{block}_fullconnect_116';g['Layers'][f'b{block}_reshape_115']['parameters']['shape']=[1,1,16,4864];g['Layers'][key]['op']='convolution';g['Layers'][key]['parameters']={**params,'weights':896}
 assert data.keys()==d.keys()
 for key,vals in data.items():
  for field,v in vals.items():
   assert v.tobytes()==d[key][field].tobytes()
   if field=='weight' and key in ['b0_fullconnect_116','b1_fullconnect_116']:assert d[key][field].shape==(1,1,4864,896)
   else:assert v.shape==d[key][field].shape
 # Independent construction: C512 Conv expanded only at exact capacity fields.
 other=json.loads((CONV512/'decoder.json').read_text());changes=next(x for x in json.loads((REPO/'research/intern-context256-20261025/graph-quantization-audit.json').read_text())['cases'] if x['name']=='prefill16')['graph_changes']
 for c in changes:
  parts=c['path'].strip('/').split('/');node=other
  for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
  key=int(parts[-1]) if isinstance(node,list) else parts[-1];expected=528 if c['before']==1296 else 512;assert node[key]==expected;node[key]=c['before']
 assert other==g,'independent capacity expansion differs'
 (OUT/'decoder.json').write_text(json.dumps(g,indent=2)+'\n');(OUT/'decoder.data').symlink_to((CONV512/'decoder.data').resolve());shutil.copy2(SRC/'decoder.quantize',OUT/'decoder.quantize');shutil.copy2(SRC/'tensor-inputmeta.yml',OUT/'tensor-inputmeta.yml')
 meta=json.loads((TASK/'fixtures-canonical/c512-m16-l2/probe-fixture.json').read_text());raw=np.fromfile(TASK/'fixtures-canonical/c512-m16-l2/probe-input.f32',np.float32);feeds={};off=0
 for key,num in meta['inputs']:
  shape=next(x['shape'] for x in meta['input_contract'] if x['name']==key);value=torch.from_numpy(raw[off:off+num].reshape(shape).copy());off+=num
  if '_past_' in key:
   x=torch.zeros(1,2,1280,64);x[:,:,:512]=value
  elif key.endswith('_mask'):
   x=torch.full((1,1,112,1296),-10000.);x[:,:,:,:512]=value[:,:,:,:512];x[:,:,:,1280:]=value[:,:,:,512:]
  else:x=value.clone()
  expected=next(v['parameters']['shape'] for v in g['Layers'].values() if v.get('op')=='input' and v['name']==key);assert list(x.shape)==expected;feeds[key]=x
 torch.set_num_threads(2);errors={}
 with torch.inference_mode():
  for mode in ['source-f32','down-weight-f16-reexpanded']:
   a=copy.deepcopy(data);b=copy.deepcopy(d)
   if mode!='source-f32':
    for block in [0,1]:
     key=f'b{block}_fullconnect_116'
     for field in ['weight','bias']:a[key][field]=a[key][field].astype(np.float16).astype(np.float32);b[key][field]=b[key][field].astype(np.float16).astype(np.float32)
   old=evaluate(original,a,feeds);new=evaluate(g,b,feeds);rows={}
   for key,x in new.items():
    y=old[key];rel=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y));torch.testing.assert_close(x,y,atol=1e-4,rtol=5e-4);assert rel<1e-5;rows[key]={'relative_l2':rel,'maxabs':float((x-y).abs().max()),'finite':bool(torch.isfinite(x).all())}
   errors[mode]=rows
 sourceconv=json.loads((SRC/'conversion.json').read_text());order=[g['Layers'][key]['name'] for key in sourceconv['input_order']];np.concatenate([feeds[k].numpy().ravel() for k in order]).tofile(OUT/'canonical-input-c1280.f32')
 state.update(state='fp32-equivalence-passed',graph_equivalence=errors,independent_two_constructions_equal=True,original_array_values_equal=True,source_hashes={n:sha(SRC/n) for n in ['decoder.json','decoder.data','decoder.quantize','export/decoder.export.data','export_nbg_unify/network_binary.nb']},canonical_input_sha256=sha(OUT/'canonical-input-c1280.f32'),source_precision='Original calibrated FP16-down exception preserved; all other dtype/quantize identical',generator_sha256=sha(Path(__file__)));save()
 cmd=json.loads((BASE512/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(BASE512),str(OUT)) for x in cmd];cmd[cmd.index('--name')+1]='opi-conv1280-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1'];state.update(state='compiling',compile_command=cmd);save();start=time.monotonic()
 with (OUT/'export.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 assert p.returncode==0,p.returncode
 conv={'first_layer':2,'fusion_width':2,'capacity':1280,'query_tokens':16,'scope':'FP16 FCL->1x1 Conv; prefill-layer2 overlay only; decode unchanged','compile_command':cmd,'exit_code':0,'seconds':time.monotonic()-start,**verify_contract(OUT,2)};(OUT/'conversion.json').write_text(json.dumps(conv,indent=2)+'\n')
 def weight(folder,key):
  code=(folder/'export/vnn_decoder.c').read_text();block=code.split('/* @'+key+':weight */',1)[1].split('/*',1)[0];m=re.search(r'VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)',block);assert m;typ,off,num=m.groups();raw=(folder/'export/decoder.export.data').read_bytes();return typ,raw[int(off):int(off)+int(num)]
 records=[]
 for key,node in original['Layers'].items():
  if node['op']!='fullconnect':continue
  ta,a=weight(SRC,key);tb,b=weight(OUT,key);assert ta==tb and a==b,(key,ta,tb);records.append({'key':key,'dtype':ta,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source_exported_weight_exact':True})
 oldmeta=json.loads((SRC/'export_nbg_unify/nbg_meta.json').read_text());newmeta=json.loads((OUT/'export_nbg_unify/nbg_meta.json').read_text());assert oldmeta==newmeta
 nb=OUT/'export_nbg_unify/network_binary.nb';state.update(state='complete',end=time.time(),compile_seconds=conv['seconds'],nb=str(nb),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size,quantizer_byte_equal=sha(OUT/'decoder.quantize')==sha(SRC/'decoder.quantize'),all_14_exported_fc_weight_dtype_and_bytes_same=records,all_external_ports_exact=True,trace=str(OUT/'export/node_trace_db.json'),board_status='prepared only; no board access or deployment');save()
 overlay={'replace_only':'prefill-layer2.nb','candidate_nb':str(nb),'candidate_sha256':sha(nb),'candidate_bytes':nb.stat().st_size,'baseline_nb_sha256':known['prefill-layer2.nb'],'capacity':1280,'query_tokens':16,'fusion_width':2,'first_layer':2,'all_other_prefill_decode_head_embedding':'unchanged from existing frozen C1280 graph set','deployment_gate':'Do not deploy before root reviews standalone C512 FP16 Conv A-B-A and approves full-model qualification','same_quantizer_bytes':True,'same_14_fc_exported_weight_bytes':True,'same_external_port_contract':True,'full_model_quality':'not tested'};(R/'deployment-overlay-c1280.json').write_text(json.dumps(overlay,indent=2)+'\n');print(json.dumps({k:v for k,v in state.items() if k in ['state','nb','nb_sha256','nb_bytes','compile_seconds']}),flush=True)
except BaseException:state.update(state='failed',end=time.time(),error=traceback.format_exc());save();raise
