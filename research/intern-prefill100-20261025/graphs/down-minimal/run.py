from pathlib import Path
import numpy as np,yaml,json,copy,subprocess,fcntl,time,hashlib,traceback,os
D=Path(__file__).resolve().parent;T=D.parent.parent
S=T/'graphs/cases/c512-m32-l2';C=T/'graphs/candidates-down-i16-conv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():(D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'status':'preparing','pid':os.getpid(),'started':time.time(),'scope':'minimal isolated down SDK diagnostic only; no board performance/quality','cases':[]};save()
try:
 lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 data=np.load(S/'decoder.data',allow_pickle=True).item()['b0_fullconnect_116'];gbase=json.loads((S/'decoder.json').read_text());conv=json.loads((C/'c512-m32-l2/decoder.json').read_text())['Layers']['b0_fullconnect_116'];qbase=yaml.safe_load((C/'c512-m32-l2/decoder.quantize').read_text())['quantize_parameters'];inputmeta=yaml.safe_load((S/'tensor-inputmeta.yml').read_text())
 cases=[(op,fl) for op in ['fc','conv'] for fl in [None,16,15,8]]
 for op,fl in cases:
  name=op+('-fp16' if fl is None else '-i16-fl'+str(fl));O=D/name;O.mkdir();shape=[32,4864] if op=='fc' else [1,1,32,4864]
  node=copy.deepcopy(gbase['Layers']['b0_fullconnect_116'] if op=='fc' else conv);node['name']='project';node['inputs']=['@input:out0']
  inp=copy.deepcopy(gbase['Layers']['b0_hidden_101']);inp.update(name='input');inp['parameters']['shape']=shape
  graph={'MetaData':{'Name':'down_probe','AcuityVersion':'6','Platform':'tensorflow','Org_Platform':'onnx'},'Layers':{'output':{'name':'output','op':'output','parameters':{},'inputs':['@project:out0'],'outputs':['out0']},'input':inp,'project':node},'quantize_info':{}}
  vals={k:v.copy() for k,v in data.items()};vals['weight']=vals['weight'].reshape((4864,896) if op=='fc' else (1,1,4864,896))
  with (O/'decoder.data').open('wb') as f:np.save(f,{'project':vals},allow_pickle=True)
  (O/'decoder.json').write_text(json.dumps(graph,indent=2)+'\n')
  if fl is None:qi=qw=qo={'qtype':'float16','quantizer':'float16'}
  else:
   qi=copy.deepcopy(qbase['@b0_reshape_115:out0']);qo=copy.deepcopy(qbase['@b0_fullconnect_116:out0']);qw=copy.deepcopy(qbase['@b0_fullconnect_116:weight']);qw['fl']=fl
  q={'version':2,'quantize_parameters':{'@input:out0':qi,'@project:weight':qw,'@project:out0':qo,'@output:out0':qo},'customized_quantize_layers':{}}
  (O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));meta=copy.deepcopy(inputmeta);port=meta['input_meta']['databases'][0]['ports'][0];port['lid']='input';port['shape']=shape;meta['input_meta']['databases'][0]['ports']=[port];(O/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
  cmd=json.loads((C/'status.json').read_text())['command'];old=str(C/'c512-m32-l2');cmd=[x.replace(old,str(O)) for x in cmd];cmd[cmd.index('--name')+1]='opi-down-min-'+name
  state.update(status='compiling',current=name);save();start=time.monotonic()
  with (O/'export.log').open('wb') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=120)
  nb=O/'export_nbg_unify/network_binary.nb';lines=(O/'export.log').read_text(errors='replace').splitlines();errors=[x for x in lines if 'CHECK STATUS' in x or 'Unsupported' in x or 'not support' in x]
  entry={'case':name,'op':op,'weight_fl':fl,'input_shape':shape,'input_quant':qi,'output_quant':qo,'returncode':r.returncode,'seconds':time.monotonic()-start,'nb_sha256':sha(nb) if nb.exists() else None,'nb_bytes':nb.stat().st_size if nb.exists() else None,'errors':errors,'source_weight_sha256':hashlib.sha256(vals['weight'].tobytes()).hexdigest(),'graph_sha256':sha(O/'decoder.json'),'quant_sha256':sha(O/'decoder.quantize')}
  (O/'result.json').write_text(json.dumps(entry,indent=2)+'\n');state['cases'].append(entry);save();print(name,r.returncode,flush=True)
 state.update(status='complete',ended=time.time());save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
