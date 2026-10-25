#!/usr/bin/env python3
"""One C512 M32 L0 FC axis3 candidate, unchanged original quantizer. No board access."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,time,traceback,uuid,re,shutil,fcntl
import numpy as np,torch
R=Path(__file__).resolve().parent;G=R.parent;TASK=G.parent;REPO=TASK.parents[1];SRC=G/'cases/c512-m32-l0';OUT=R/'fc-axis3-c512-m32-l0';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
sys.path.insert(0,str(G/'candidates-fp16'));from graph_reference_conv import evaluate
state={'state':'preparing','start':time.time(),'source':str(SRC),'candidate':str(OUT),'scope':'single fused2 layers0/1 C512 M32; 14 original FC ops retained, private rank4 inputs and axis3 only; no board'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():(R/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def block_for(code,key):
 matches=[body for names,body in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',code,re.S) if '@'+key in names.split()]
 assert len(matches)==1,(key,len(matches))
 return matches[0]
def quant_sig(code,b):
 q=re.search(r'attr.dtype.qnt_type = (\w+);',b);ret={'qnt_type':q[1] if q else None}
 for field in ['fl','scale_dim','zero_points_dim']:
  m=re.search(r'attr.dtype.'+field+r' = (-?\d+);',b)
  if m:ret[field]=int(m[1])
 for field,typ in [('scales','<f4'),('zero_points','<i4')]:
  m=re.search(r'attr.dtype.'+field+r' = (\w+);',b)
  if m:
   x=re.search(r'\b'+m[1]+r'\[\]\s*=\s*\{([^}]+)\}',code);assert x,m[1]
   a=np.asarray([float(v.strip().rstrip('f')) for v in x[1].split(',') if v.strip()],dtype=typ)
   ret[field+'_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
 return ret
def constant(folder,key):
 code=(folder/'export/vnn_decoder.c').read_text();b=block_for(code,key)
 m=re.search(r'VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)',b);assert m,key
 typ,off,num=m.groups();raw=(folder/'export/decoder.export.data').read_bytes()[int(off):int(off)+int(num)]
 q=quant_sig(code,b);q['dtype']=typ
 axis=re.search(r'attr.dtype.channel_dim = (\d+);',b)
 dims={int(k):int(v) for k,v in re.findall(r'attr.size\[(\d+)\] = (\d+);',b)}
 return raw,q,(int(axis[1]) if axis else None),dims
save()
try:
 base=next(x for x in json.loads((G/'deployment-manifest.json').read_text())['cases'] if x['id']=='c512-m32-l0');assert sha(SRC/'export_nbg_unify/network_binary.nb')==base['nb_sha256']
 original=json.loads((SRC/'decoder.json').read_text());g=copy.deepcopy(original);data=np.load(SRC/'decoder.data',allow_pickle=True).item();d=copy.deepcopy(data)
 keys=[k for k,v in original['Layers'].items() if v['op']=='fullconnect'];assert len(keys)==14
 changes=[]
 for key in keys:
  node=g['Layers'][key];w=data[key]['weight'];assert w.ndim==2;ni,no=w.shape;inp=node['inputs'][0];pre=inp.split(':')[0][1:]
  assert original['Layers'][pre]['op']=='reshape' and original['Layers'][pre]['parameters']['shape']==[-1,ni]
  users=[k for k,v in original['Layers'].items() if inp in v['inputs']];assert users==[key],(key,users)
  g['Layers'][pre]['parameters']['shape']=[1,1,32,ni];assert node['parameters']['axis']==1;node['parameters']['axis']=3
  changes.append({'key':key,'private_input_reshape':pre,'input_features':ni,'output_features':no,'op':'fullconnect','axis_before_after':[1,3],'source_weight_bytes_equal':d[key]['weight'].tobytes()==w.tobytes()})
 for key,vals in data.items():
  for field,v in vals.items():assert v.tobytes()==d[key][field].tobytes(),(key,field)
 (OUT/'decoder.json').write_text(json.dumps(g,indent=2)+'\n')
 (OUT/'decoder.data').symlink_to((SRC/'decoder.data').resolve());assert sha(SRC/'decoder.data')==sha(OUT/'decoder.data')
 shutil.copy2(SRC/'decoder.quantize',OUT/'decoder.quantize');shutil.copy2(SRC/'tensor-inputmeta.yml',OUT/'tensor-inputmeta.yml');assert sha(SRC/'decoder.quantize')==sha(OUT/'decoder.quantize')
 fixture=TASK/'fixtures-canonical/c512-m32-l0';meta=json.loads((fixture/'probe-fixture.json').read_text());raw=np.fromfile(fixture/'probe-input.f32',np.float32);feeds={};off=0
 for key,num in meta['inputs']:
  shape=next(x['shape'] for x in meta['input_contract'] if x['name']==key);feeds[key]=torch.from_numpy(raw[off:off+num].reshape(shape).copy());off+=num
 assert off==raw.size and meta['query_tokens']==32 and meta['capacity']==512 and meta['first_layer']==0
 assert sha(fixture/'probe-input.f32')==meta['input_sha256'];torch.set_num_threads(2);errors={}
 for mode in ['source-f32','actual-source-exported-fc-dequantized']:
  a=copy.deepcopy(data);b=copy.deepcopy(d)
  if mode!='source-f32':
   code=(SRC/'export/vnn_decoder.c').read_text()
   for key in keys:
    for field in ['weight','bias']:
     rawc,q,axis,dims=constant(SRC,key+':'+field);assert q['dtype']==('INT8' if field=='weight' else 'INT64')
     vals=np.frombuffer(rawc,np.dtype('i1' if field=='weight' else '<i8')).astype(np.float64)
     if field=='weight':vals=vals.reshape(data[key]['weight'].shape[::-1]).T
     bl=block_for(code,key+':'+field);m=re.search(r'attr.dtype.scales = (\w+);',bl)
     if m:
      arr=re.search(r'\b'+m[1]+r'\[\]\s*=\s*\{([^}]+)\}',code);scale=np.asarray([float(v.strip().rstrip('f')) for v in arr[1].split(',') if v.strip()],np.float32);vals*=scale
     else:vals*=2.**(-q['fl'])
     vals=vals.astype(np.float32);a[key][field]=vals.reshape(data[key][field].shape);b[key][field]=vals.reshape(d[key][field].shape)
  with torch.inference_mode():old=evaluate(original,a,feeds);new=evaluate(g,b,feeds)
  rows={}
  for key,x in new.items():
   y=old[key];rel=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y));torch.testing.assert_close(x,y,atol=1e-4,rtol=5e-4);assert rel<1e-5;rows[key]={'relative_l2':rel,'maxabs':float((x-y).abs().max()),'finite':bool(torch.isfinite(x).all())}
  errors[mode]=rows
 state.update(state='fp32-equivalence-passed',changes=changes,graph_equivalence=errors,source_hashes={n:sha(SRC/n) for n in ['decoder.json','decoder.data','decoder.quantize','export/decoder.export.data','export_nbg_unify/network_binary.nb']},input=str(fixture/'probe-input.f32'),input_sha256=meta['input_sha256'],reference=str(fixture/'probe-reference.f32'),reference_sha256=meta['reference_sha256'],generator_sha256=sha(__file__));save()
 cmd=json.loads((SRC/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(SRC),str(OUT)) for x in cmd];cmd[cmd.index('--name')+1]='opi-fcaxis3-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1'];state.update(state='waiting-compile-lock',compile_command=cmd,compile_lock='/tmp/opi-prefill-compile.lock');save()
 with open('/tmp/opi-prefill-compile.lock','a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX);state.update(state='compiling');save();start=time.monotonic()
  with (OUT/'export.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  compile_seconds=time.monotonic()-start
  fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
 assert p.returncode==0,p.returncode
 conv={'first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':32,'scope':'14 FC retained; axis3/private rank4 input; single block only','compile_command':cmd,'exit_code':0,'seconds':compile_seconds,**verify_contract(OUT,2)};(OUT/'conversion.json').write_text(json.dumps(conv,indent=2)+'\n')
 records=[]
 oldcode=(SRC/'export/vnn_decoder.c').read_text();newcode=(OUT/'export/vnn_decoder.c').read_text()
 for key in keys:
  rec={'key':key}
  for field in ['weight','bias']:
   a,qa,axa,da=constant(SRC,key+':'+field);b,qb,axb,db=constant(OUT,key+':'+field)
   assert qa==qb and a==b,(key,field,qa,qb);assert qa['dtype']==('INT8' if field=='weight' else 'INT64')
   if axa is not None:assert da[axa]==db[axb],(key,field,da,db)
   rec[field]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'quantization':qb,'source_dtype_bytes_scales_exact':True,'channel_axis_before_after':[axa,axb]}
  for port in [key+':out0',g['Layers'][key]['inputs'][0][1:]]:
   a=block_for(oldcode,port);b=block_for(newcode,port);qa=quant_sig(oldcode,a);qb=quant_sig(newcode,b)
   ta=re.search(r'VSI_NN_TYPE_(\w+)\)',a)[1];tb=re.search(r'VSI_NN_TYPE_(\w+)\)',b)[1];assert qa==qb and ta==tb=='INT16',(port,qa,qb,ta,tb)
  rec['input_output_I16_quantization_exact']=True;records.append(rec)
 whole_payload_same=sha(SRC/'export/decoder.export.data')==sha(OUT/'export/decoder.export.data');assert whole_payload_same,'constant payload differs'
 ops=re.findall(r'NEW_VXNODE\(node\[\d+\], VSI_NN_OP_(\w+),',newcode);lowering={k:ops.count(k) for k in sorted(set(ops))};state.update(actual_lowering_counts=lowering,full_exported_constant_payload_same=True,exported_payload_sha256=sha(OUT/'export/decoder.export.data'));save()
 oldmeta=json.loads((SRC/'export_nbg_unify/nbg_meta.json').read_text());newmeta=json.loads((OUT/'export_nbg_unify/nbg_meta.json').read_text());assert oldmeta==newmeta
 nb=OUT/'export_nbg_unify/network_binary.nb';state.update(state='complete',end=time.time(),compile_seconds=conv['seconds'],nb=str(nb),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size,quantizer_byte_equal=True,all_14_exported_weights_biases_quantization=records,all_external_ports_exact=True,trace=str(OUT/'export/node_trace_db.json'),board_status='prepared only; no board access or deployment');save();print(json.dumps({k:v for k,v in state.items() if k in ['state','nb','nb_sha256','nb_bytes','compile_seconds']}),flush=True)
except BaseException:state.update(state='failed',end=time.time(),error=traceback.format_exc());save();raise
