from pathlib import Path
import os,sys,json,hashlib,copy,shutil,subprocess,fcntl,time,traceback,uuid
import numpy as np,yaml
R=Path(__file__).resolve().parent;T=R.parent;rows=int(sys.argv[1]);assert rows in [8,32];D=R/('full-c448-m'+str(rows));D.mkdir(exist_ok=False)
S=T/'c448-qualified-r3'/('original'+str(rows));Q=T/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def write(f,x):f.write_text(json.dumps(x,indent=2)+'\n')
st={'status':'preparing','pid':os.getpid(),'start':time.time()}
def save():write(D/'status.json',st)
save()
try:
 srcsha={n:sha(S/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','export_nbg_unify/network_binary.nb']}
 g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();og=copy.deepcopy(g);oq=copy.deepcopy(q);od=copy.deepcopy(data)
 iq=yaml.safe_load((Q/'decoder.quantize').read_text())['quantize_parameters'];ig=json.loads((Q/'decoder.json').read_text());selection=json.loads((R/'ANALYSIS.json').read_text());added=[];modified={};qchanged=[];splits=[]
 for b in range(2):
  ids=next(x for x in selection['layers'][b]['splits'] if x['fp16_channels']==8)['channel_ids'];assert len(set(ids))==8
  fc=f'b{b}_fullconnect_116';src=f'@b{b}_reshape_115:out0';cv=f'down_i16_{2000+b}';slow=f'outlier_slowfc_{8100+b}';cat=f'outlier_concat_{8200+b}';summ=f'outlier_sum_{8300+b}';restore=f'b{b}_reshape_117'
  assert g['Layers'][fc]['inputs']==[src] and g['Layers'][restore]['inputs']==['@'+fc+':out0'];assert np.count_nonzero(data[fc]['bias'])==0
  def add(n,op,inputs,params):
   assert n not in g['Layers'];g['Layers'][n]={'name':n,'op':op,'inputs':inputs,'parameters':params,'outputs':['out0']};q['quantize_parameters']['@'+n+':out0']={'qtype':'float16','quantizer':'float16'};added.append(n);qchanged.append('@'+n+':out0')
  slices=[]
  for i,ch in enumerate(ids):
   n=f'outlier_slice_{8400+b*10+i}';add(n,'slice',[src],{'begin':[0,ch],'size':[rows,1]});slices.append('@'+n+':out0')
  add(cat,'concat',slices,{'dim':1});add(slow,'fullconnect',['@'+cat+':out0'],copy.deepcopy(g['Layers'][fc]['parameters']));add(summ,'add',['@'+fc+':out0','@'+slow+':out0'],{})
  g['Layers'][cv]=copy.deepcopy(ig['Layers'][cv]);added.append(cv);q['quantize_parameters']['@'+cv+':out0']=copy.deepcopy(iq['@'+cv+':out0']);qchanged.append('@'+cv+':out0')
  for n in [fc,restore]:modified[n]=copy.deepcopy(g['Layers'][n])
  g['Layers'][fc]['inputs']=['@'+cv+':out0'];g['Layers'][restore]['inputs']=['@'+summ+':out0']
  data[slow]={'weight':data[fc]['weight'][ids,:].copy(),'bias':data[fc]['bias'].copy()};data[fc]['weight'][ids,:]=0
  w=data[fc]['weight'].copy();w[ids,:]+=data[slow]['weight'];assert np.array_equal(w,od[fc]['weight'])
  for kind in ['weight']+(['bias'] if '@'+fc+':bias' in oq['quantize_parameters'] else []):
   q['quantize_parameters']['@'+slow+':'+kind]=copy.deepcopy(oq['quantize_parameters']['@'+fc+':'+kind]);qchanged.append('@'+slow+':'+kind)
  q['quantize_parameters']['@'+fc+':weight']=copy.deepcopy(iq['@'+fc+':weight']);qchanged.append('@'+fc+':weight')
  # Original output stays FP16. The integer FC output then sums the exact-channel FP16 branch.
  assert q['quantize_parameters']['@'+fc+':out0']['qtype']=='float16'
  splits.append({'layer':b+2,'indices':ids,'source_shape':list(od[fc]['weight'].shape),'slow_shape':list(data[slow]['weight'].shape),'weight_reconstruction_exact':True})
 undo=copy.deepcopy(g)
 for n in added:del undo['Layers'][n]
 for n,v in modified.items():undo['Layers'][n]=v
 assert undo==og
 assert all(q['quantize_parameters'][k]==v for k,v in oq['quantize_parameters'].items() if k not in qchanged)
 for k in od:
  if k in ['b0_fullconnect_116','b1_fullconnect_116']:continue
  assert set(data[k])==set(od[k]) and all(np.array_equal(data[k][a],od[k][a]) for a in od[k]),k
 write(D/'decoder.json',g);(D/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
 with (D/'decoder.data').open('wb') as f:np.save(f,data,allow_pickle=True)
 shutil.copy2(S/'tensor-inputmeta.yml',D/'tensor-inputmeta.yml')
 audit={'source_sha256':srcsha,'selection_sha256':sha(R/'ANALYSIS.json'),'calibration_case':selection['calibration_case'],'heldout_case':'s128-d256-0','splits':splits,'added_nodes':added,'modified_nodes_before':modified,'quantizer_changed_keys':qchanged,'scope':'C448 L2/L3 split down candidate, M'+str(rows)+'; not full-model validated'};write(D/'audit.json',audit)
 SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  st['status']='waiting_compile_lock';save();fcntl.flock(lock,fcntl.LOCK_EX)
  assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
  B=T/'graphs/cases/c512-m32-l2';cmd=[x.replace(str(B),str(D)) for x in json.loads((B/'conversion.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-outlier-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
  cmd[3:3]=['-v',str(R)+':'+str(R)];st.update(status='compiling',command=cmd);save()
  with (D/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  assert z.returncode==0,z.returncode
 assert all(sha(S/n)==h for n,h in srcsha.items())
 nb=D/'export_nbg_unify/network_binary.nb';st.update(status='compiled-unvalidated',ended=time.time(),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size);save()
except BaseException:st.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
