from pathlib import Path
import numpy as np,yaml,json,copy,hashlib,subprocess,fcntl,time,os,traceback,sys,uuid,shutil
ROOT=Path(__file__).resolve().parent;T=ROOT.parent;R=T.parents[1];D=ROOT/"attempt2";D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def qt(x):return {k:v for k,v in x.items() if k not in ['max_value','min_value']}
s={'status':'preflight','pid':os.getpid(),'started':time.time(),'cases':[]}
def save():write(D/'status.json',s)
save()
try:
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);sys.path.insert(0,str(R/'npu/kv-cache'));from fused_contract import verify_contract
  audits=[]
  for layer in range(0,24,2):
   P=T/'compact-m64-r4/graphs'/f'l{layer}';L=json.loads((P/'decoder.json').read_text())['Layers'];q=yaml.safe_load((P/'decoder.quantize').read_text())['quantize_parameters']
   for b in range(2):
    a=f'b{b}_fullconnect_125';z=f'b{b}_fullconnect_131';ia=L[a]['inputs'][0];ib=L[z]['inputs'][0];ra=L[ia[1:].split(':')[0]];rb=L[ib[1:].split(':')[0]]
    same_source=ra['inputs']==rb['inputs'] and ra['parameters']==rb['parameters']
    match={'input':qt(q[ia])==qt(q[ib]),**{suffix:qt(q['@'+a+':'+suffix])==qt(q['@'+z+':'+suffix]) for suffix in ['out0','weight','bias']}}
    audits.append({'layer':layer+b,'graph_sha256':sha(P/'decoder.json'),'quantize_sha256':sha(P/'decoder.quantize'),'same_source':same_source,'matching_quantizers':match,'eligible_without_rescaling':same_source and all(match.values())})
  write(D/'PREFLIGHT.json',{'scope':'24-layer static graph audit; no speed or numerical equivalence claim','layers':audits})
  for name,S,rows,past in [('m32-p448',T/'down-outlier-split-r2/full-c448-m32',32,448),('m64-p128',T/'compact-m64-r4/graphs/l2',64,128)]:
   P=D/name;P.mkdir(exist_ok=False);g=json.loads((S/'decoder.json').read_text());og=copy.deepcopy(g);q=yaml.safe_load((S/'decoder.quantize').read_text());oq=copy.deepcopy(q);params=q['quantize_parameters'];data=np.load(S/'decoder.data',allow_pickle=True).item()
   frozen={n:sha(S/n) for n in ['decoder.json','decoder.data','decoder.quantize','export_nbg_unify/network_binary.nb']}
   a='b1_fullconnect_125';z='b1_fullconnect_131';key='b1_gateup_fused_12000';L=g['Layers'];na=L[a];nz=L[z]
   assert na['op']==nz['op']=='fullconnect' and na['parameters']==nz['parameters'] and na['parameters']['weights']==4864
   ia=na['inputs'][0];ib=nz['inputs'][0];ra=L[ia[1:].split(':')[0]];rb=L[ib[1:].split(':')[0]];assert ra['inputs']==rb['inputs'] and ra['parameters']==rb['parameters'];assert qt(params[ia])==qt(params[ib])
   for suffix in ['out0','weight','bias']:assert qt(params['@'+a+':'+suffix])==qt(params['@'+z+':'+suffix])
   original_arrays={k:{n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in d.items()} for k,d in data.items()}
   wa,wb=data[a]['weight'],data[z]['weight'];ba,bb=data[a]['bias'],data[z]['bias'];assert wa.shape==wb.shape==(896,4864) and ba.shape==bb.shape==(4864,)
   fused=copy.deepcopy(na);fused['name']=key;fused['parameters']['weights']=9728;L[key]=fused
   data[key]={'weight':np.concatenate([wa,wb],axis=1),'bias':np.concatenate([ba,bb])}
   assert np.array_equal(data[key]['weight'][:,:4864],wa) and np.array_equal(data[key]['weight'][:,4864:],wb) and np.array_equal(data[key]['bias'][:4864],ba) and np.array_equal(data[key]['bias'][4864:],bb)
   for suffix in ['out0','weight','bias']:
    v=copy.deepcopy(params['@'+a+':'+suffix]);w=params['@'+z+':'+suffix]
    if 'max_value' in v:v['max_value']=max(v['max_value'],w['max_value']);v['min_value']=min(v['min_value'],w['min_value'])
    params['@'+key+':'+suffix]=v
   for index,old in enumerate([a,z]):
    L[old]={'name':og['Layers'][old]['name']+'_slice','op':'slice','parameters':{'begin':[0,index*4864],'size':[rows,4864]},'inputs':['@'+key+':out0'],'outputs':['out0']}
    del data[old]
    for suffix in ['weight','bias']:del params['@'+old+':'+suffix]
   assert all(L[k]==v for k,v in og['Layers'].items() if k not in [a,z])
   assert all(params[k]==v for k,v in oq['quantize_parameters'].items() if k not in ['@'+n+':'+s for n in [a,z] for s in ['weight','bias']])
   assert all(hashlib.sha256(data[k][n].tobytes()).hexdigest()==h for k,d in original_arrays.items() if k not in [a,z] for n,h in d.items())
   assert key not in og['Layers'];assert all(k.rsplit('_',1)[-1].isdigit() for k in L)
   write(P/'decoder.json',g);(P/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
   with (P/'decoder.data').open('wb') as f:np.save(f,data,allow_pickle=True)
   shutil.copy2(S/'tensor-inputmeta.yml',P/'tensor-inputmeta.yml')
   base=T/'compact-m64-r4/graphs/l0';cmd=json.loads((base/'build.json').read_text())['compile_command'];cmd[cmd.index('-w')+1]=str(P);cmd[cmd.index('--name')+1]='opi-gateup-r9-'+uuid.uuid4().hex[:8]
   s.update(status='compiling',current=name);save()
   with (P/'compile.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=360)
   assert r.returncode==0,r.returncode
   assert all(sha(S/n)==h for n,h in frozen.items());ports=verify_contract(P,2);oldports=verify_contract(S,2);assert ports['input_order']==oldports['input_order'] and ports['output_order']==oldports['output_order']
   build={'name':name,'parent':str(S),'source_sha256':frozen,'rows':rows,'past':past,'fused_layer':3,'nb_sha256':sha(P/'export_nbg_unify/network_binary.nb'),'nb_bytes':(P/'export_nbg_unify/network_binary.nb').stat().st_size,'ports':ports,'source_float_weight_values_preserved':True,'quantizer_type_scale_rounding_preserved':True,'quantization_minmax_metadata_union':True,'original_fc_keys_now_slice': [a,z],'command':cmd,'status':'compiled-unvalidated'}
   write(P/'build.json',build);s['cases'].append(build);save();print(name,build['nb_sha256'],flush=True)
  s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
