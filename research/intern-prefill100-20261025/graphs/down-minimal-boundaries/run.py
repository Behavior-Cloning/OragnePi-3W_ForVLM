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
 cases=[(op,16,bridge) for op in ['fc','conv'] for bridge in ['input','output']]
 for op,fl,bridge in cases:
  name=op+'-i16-'+bridge+'-bridge';O=D/name;O.mkdir();shape=[32,4864] if op=='fc' else [1,1,32,4864]
  node=copy.deepcopy(gbase['Layers']['b0_fullconnect_116'] if op=='fc' else conv);node['name']='project';node['inputs']=['@input_1:out0']
  inp=copy.deepcopy(gbase['Layers']['b0_hidden_101']);inp.update(name='input');inp['parameters']['shape']=shape
  graph={'MetaData':{'Name':'down_probe','AcuityVersion':'6','Platform':'tensorflow','Org_Platform':'onnx'},'Layers':{'output_0':{'name':'output','op':'output','parameters':{},'inputs':['@project_2:out0'],'outputs':['out0']},'input_1':inp,'project_2':node},'quantize_info':{}}
  graph['Layers']['input_1']['parameters']['shape']=[1,32,4864]
  graph['Layers']['reshape_in_3']={'name':'reshape_in','op':'reshape','parameters':{'shape':shape},'inputs':['@input_1:out0'],'outputs':['out0']}
  graph['Layers']['project_2']['inputs']=['@reshape_in_3:out0']
  graph['Layers']['reshape_out_4']={'name':'reshape_out','op':'reshape','parameters':{'shape':[1,32,896]},'inputs':['@project_2:out0'],'outputs':['out0']}
  graph['Layers']['output_0']['inputs']=['@reshape_out_4:out0']
  vals={k:v.copy() for k,v in data.items()};vals['weight']=vals['weight'].reshape((4864,896) if op=='fc' else (1,1,4864,896))
  with (O/'decoder.data').open('wb') as f:np.save(f,{'project_2':vals},allow_pickle=True)
  (O/'decoder.json').write_text(json.dumps(graph,indent=2)+'\n')
  if fl is None:qi=qw=qo={'qtype':'float16','quantizer':'float16'}
  else:
   qi=copy.deepcopy(qbase['@b0_reshape_115:out0']);qo=copy.deepcopy(qbase['@b0_fullconnect_116:out0']);qw=copy.deepcopy(qbase['@b0_fullconnect_116:weight']);qw['fl']=fl
  q={'version':2,'quantize_parameters':{'@input_1:out0':qi,'@project_2:weight':qw,'@project_2:out0':qo,'@output_0:out0':qo},'customized_quantize_layers':{}}
  q['quantize_parameters'].update({'@reshape_in_3:out0':qi,'@input_1:out0':({'qtype':'float16','quantizer':'float16'} if bridge=='input' else qi),'@reshape_out_4:out0':({'qtype':'float16','quantizer':'float16'} if bridge=='output' else qo),'@output_0:out0':({'qtype':'float16','quantizer':'float16'} if bridge=='output' else qo)})
  (O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));meta=copy.deepcopy(inputmeta);port=meta['input_meta']['databases'][0]['ports'][0];port['lid']='input_1';port['shape']=[1,32,4864];meta['input_meta']['databases'][0]['ports']=[port];(O/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
  cmd=json.loads((C/'status.json').read_text())['command'];old=str(C/'c512-m32-l2');cmd=[x.replace(old,str(O)) for x in cmd];cmd[cmd.index('--name')+1]='opi-down-boundary-'+name
  state.update(status='compiling',current=name);save();start=time.monotonic()
  with (O/'export.log').open('wb') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=120)
  nb=O/'export_nbg_unify/network_binary.nb';lines=(O/'export.log').read_text(errors='replace').splitlines();errors=[x for x in lines if 'CHECK STATUS' in x or 'Unsupported' in x or 'not support' in x]
  entry={'case':name,'op':op,'weight_fl':fl,'input_shape':shape,'input_quant':qi,'output_quant':qo,'returncode':r.returncode,'seconds':time.monotonic()-start,'nb_sha256':sha(nb) if nb.exists() else None,'nb_bytes':nb.stat().st_size if nb.exists() else None,'errors':errors,'source_weight_sha256':hashlib.sha256(vals['weight'].tobytes()).hexdigest(),'graph_sha256':sha(O/'decoder.json'),'quant_sha256':sha(O/'decoder.quantize')}
  (O/'result.json').write_text(json.dumps(entry,indent=2)+'\n');state['cases'].append(entry);save();print(name,r.returncode,flush=True)
  if fl is None and (r.returncode!=0 or not nb.exists()):raise RuntimeError('FP16 control failed; stop matrix, inspect before continuing')
 state.update(status='complete',ended=time.time());save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
