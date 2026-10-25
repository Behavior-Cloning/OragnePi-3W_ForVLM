"""Compile compact-history + row-normalized down bridge screens; no full-model claim."""
from pathlib import Path
import copy,fcntl,hashlib,json,shutil,subprocess,time,traceback,uuid
import numpy as np,yaml
G=Path(__file__).resolve().parent;D=G/'compact-row-down-screen';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
st={'state':'preparing','started':time.time(),'cases':[]}
def save():write(D/'status.json',st)
save()
try:
 rs=G/'down-row-scale-c512-m32-l2';rg=json.loads((rs/'decoder.json').read_text());rq=yaml.safe_load((rs/'decoder.quantize').read_text());rd=np.load(rs/'decoder.data',allow_pickle=True).item()
 rh={n:sha(rs/n) for n in ['decoder.json','decoder.quantize','decoder.data','source-audit.json']}
 SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  st['state']='waiting_sdk_lock';save();fcntl.flock(lock,fcntl.LOCK_EX)
  assert shutil.disk_usage('/mnt/f').free>8*1024**3
  assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
  for M,mode in [(64,'plain'),(64,'row'),(256,'row')]:
   S=G/'cases'/f'c512-m{M}-l2';O=D/f'm{M}-{mode}';O.mkdir();st.update(state='preparing',current=O.name);save()
   frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','tensor-inputmeta.yml','conversion.json']}
   g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());og=copy.deepcopy(g);oq=copy.deepcopy(q);data=np.load(S/'decoder.data',allow_pickle=True).item();geometry=[]
   def ch(k,param,axis,old,new):
    x=g['Layers'][k]['parameters'][param];assert x[axis]==old,(k,x);x[axis]=new;geometry.append({'node':k,'param':param,'axis':axis,'old':old,'new':new})
   for b in range(2):
    ch(f'b{b}_past_k_102','shape',2,512,128);ch(f'b{b}_past_v_103','shape',2,512,128);ch(f'b{b}_mask_104','shape',3,512+M,128+M);ch(f'b{b}_split_probs_past_11004','size',3,512,128);ch(f'b{b}_split_probs_new_11005','begin',3,512,128)
   compact=copy.deepcopy(g);added=[];modified={};qchanged={}
   if mode=='row':
    for b in range(2):
     fc=f'b{b}_fullconnect_116';cv=f'down_i16_{2000+b}';restore=f'b{b}_reshape_117'
     assert np.array_equal(data[fc]['weight'],rd[fc]['weight']) and np.array_equal(data[fc]['bias'],rd[fc]['bias']) and np.count_nonzero(data[fc]['bias'])==0
     assert g['Layers'][fc]['inputs']==[f'@b{b}_reshape_115:out0']
     for n in [cv]+[k for k in rg['Layers'] if k.startswith('row_') and int(k.rsplit('_',1)[1])//10==(700+b)]:
      assert n not in g['Layers'];g['Layers'][n]=copy.deepcopy(rg['Layers'][n]);added.append(n)
      key='@'+n+':out0';assert key not in q['quantize_parameters'];q['quantize_parameters'][key]=copy.deepcopy(rq['quantize_parameters'][key]);qchanged[key]={'before':None,'after':q['quantize_parameters'][key]}
     for n in [fc,restore]:
      modified[n]=copy.deepcopy(g['Layers'][n]);g['Layers'][n]['inputs']=copy.deepcopy(rg['Layers'][n]['inputs'])
     for key in ['@'+fc+':weight','@'+fc+':out0']:
      qchanged[key]={'before':copy.deepcopy(q['quantize_parameters'][key]),'after':copy.deepcopy(rq['quantize_parameters'][key])};q['quantize_parameters'][key]=copy.deepcopy(rq['quantize_parameters'][key])
    assert len(added)==12 and len(modified)==4
   undo=copy.deepcopy(g)
   for n in added:del undo['Layers'][n]
   for n,v in modified.items():undo['Layers'][n]=v
   assert undo==compact
   for e in geometry:undo['Layers'][e['node']]['parameters'][e['param']][e['axis']]=e['old']
   assert undo==og
   assert all(q['quantize_parameters'][k]==v for k,v in oq['quantize_parameters'].items() if k not in qchanged)
   write(O/'decoder.json',g);(O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(O/'decoder.data').symlink_to((S/'decoder.data').resolve())
   meta=yaml.safe_load((S/'tensor-inputmeta.yml').read_text())
   for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
   (O/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
   write(O/'transform.json',{'source_sha256':frozen,'row_reference_sha256':rh,'geometry':geometry,'added_nodes':added,'modified_nodes_before':modified,'quantization_changes':qchanged,'source_weights_identical':True,'scope':'L2/L3 only; compact capacity128 retains exactly history128; no truncation of valid history. Different M is a performance screen, not whole request. Later positions require larger capacity. Row scale changes precision; not qualified.'})
   cmd=[x.replace(str(S),str(O)) for x in json.loads((S/'conversion.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-compact-row-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
   st.update(state='compiling',command=cmd);save()
   with (O/'compile.log').open('w') as log:
    try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
   assert z.returncode==0,z.returncode
   assert all(sha(S/n)==h for n,h in frozen.items()) and all(sha(rs/n)==h for n,h in rh.items())
   nb=O/'export_nbg_unify/network_binary.nb';r={'name':O.name,'capacity':128,'rows':M,'first_layer':2,'mode':mode,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'command':cmd};write(O/'conversion.json',r);st['cases'].append(r);save()
  st.update(state='complete',ended=time.time());save()
except BaseException:st.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
