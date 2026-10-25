"""Native SDK RMSNorm lowering screen, not a numerically qualified replacement."""
from pathlib import Path
import copy,fcntl,hashlib,json,subprocess,time,traceback,uuid
import numpy as np,yaml
G=Path(__file__).resolve().parent;S=G/'cases/c512-m32-l0';D=G/'native-rms-c512-m32-l0';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
st={'state':'preparing','started':time.time()}
def save():write(D/'status.json',st)
save()
try:
 frozen={n:sha(S/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml']}
 graph=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();oldg=copy.deepcopy(graph);oldq=copy.deepcopy(q);oldd=copy.deepcopy(data);changes=[]
 for block in range(2):
  for which,outstem,divstem,gammastem,epsstem in [(1,'Mul_4_output_0_28','Div_output_0_47','norm1_39','Constant_1_output_0_74'),(2,'Mul_13_output_0_36','Div_3_output_0_52','norm2_46','Constant_43_output_0_80')]:
   out=f'b{block}_{outstem}';source=graph['Layers'][f'b{block}_{divstem}']['inputs'][0];stop=source[1:].split(':')[0];gamma=f'b{block}_{gammastem}';epsnode=f'b{block}_{epsstem}'
   scale=np.asarray(data[gamma]['data']);assert scale.shape==(1,1,896) and np.isfinite(scale).all()
   epsroot=float(np.asarray(data[epsnode]['data']).reshape(-1)[0]);assert abs(epsroot-.001)<1e-9
   found=set()
   def visit(k):
    if k==stop or k in found:return
    found.add(k)
    for v in graph['Layers'][k].get('inputs',[]):visit(v[1:].split(':')[0])
   visit(out);assert gamma in found and epsnode in found and stop not in found
   for k,node in graph['Layers'].items():
    if k not in found:
     assert not any(v[1:].split(':')[0] in found-{out} for v in node.get('inputs',[])),('shared internal RMS node',k)
   oldnodes={k:graph['Layers'][k] for k in sorted(found)}
   old_quant={k:v for k,v in q['quantize_parameters'].items() if k.startswith('@') and k[1:].split(':')[0] in found}
   for k in found:
    del graph['Layers'][k]
    data.pop(k,None)
   graph['Layers'][out]={'name':out,'op':'rmsnormalize','parameters':{'axis':-1,'eps':epsroot*epsroot},'inputs':[source],'outputs':['out0']}
   data[out]={'scale':scale.reshape(896).copy()}
   for k in old_quant:del q['quantize_parameters'][k]
   q['quantize_parameters']['@'+out+':out0']=copy.deepcopy(old_quant['@'+out+':out0'])
   q['quantize_parameters']['@'+out+':scale']={'qtype':'float16','quantizer':'float16'}
   changes.append({'block':block,'norm':which,'output':out,'source':source,'epsilon':epsroot*epsroot,'original_gamma_node':gamma,'gamma_raw_sha256':hashlib.sha256(scale.tobytes()).hexdigest(),'removed_nodes':oldnodes,'removed_quant':old_quant})
 # Check unchanged coefficients and all surviving external graph nodes.
 changed={c['output'] for c in changes};removed=set(k for c in changes for k in c['removed_nodes'])
 assert all(graph['Layers'][k]==v for k,v in oldg['Layers'].items() if k not in removed)
 assert all(np.array_equal(data[k][sub],v) for k,ds in oldd.items() if k not in removed for sub,v in ds.items())
 write(D/'decoder.json',graph);(D/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
 with (D/'decoder.data').open('wb') as f:np.save(f,data,allow_pickle=True)
 (D/'tensor-inputmeta.yml').write_bytes((S/'tensor-inputmeta.yml').read_bytes())
 write(D/'transform.json',{'source_sha256':frozen,'changes':changes,'nodes_before':len(oldg['Layers']),'nodes_after':len(graph['Layers']),'scope':'Replace stable max-scaled RMS decomposition with native RMSNormalize(axis=-1, eps~1e-6); same gamma, same output quantizer. Mathematical equivalence in real arithmetic only; FP16 reduction stability and SDK lowering unknown. No precision approval.'})
 st['state']='waiting_sdk_lock';save()
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  sdk='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3';assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==sdk
  cmd=[x.replace(str(S),str(D)) for x in json.loads((S/'conversion.json').read_text())['compile_command']]
  cmd[cmd.index('--name')+1]='opi-native-rms-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=sdk
  st.update(state='compiling',command=cmd);save()
  with (D/'compile.log').open('w') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
   except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
  assert rc==0,rc
  assert all(sha(S/n)==h for n,h in frozen.items())
  nb=D/'export_nbg_unify/network_binary.nb';st.update(state='complete',ended=time.time(),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size);save()
except BaseException:st.update(state='failed',ended=time.time(),error=traceback.format_exc());save();raise
