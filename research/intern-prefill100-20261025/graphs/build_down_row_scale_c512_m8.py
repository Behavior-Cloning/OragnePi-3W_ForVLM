"""Private per-row dynamic normalization candidate; not a quality-approved graph."""
from pathlib import Path
import copy,fcntl,hashlib,json,shutil,subprocess,time,traceback,uuid,re
import numpy as np,yaml
G=Path(__file__).resolve().parent;S=G.parent/'capacity512-range16/cases/prefill8/l2';D=G/'down-row-scale-c512-m8-l2';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
st={'state':'waiting_sdk_lock','started':time.time()}
def save():write(D/'status.json',st)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 sdk='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3';assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==sdk
 frozen={n:sha(S/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
 g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();og=copy.deepcopy(g);oq=copy.deepcopy(q);added=[];rewires=[]
 for b in range(2):
  fc=f'b{b}_fullconnect_116';converter=f'down_i16_{2000+b}';source=f'@b{b}_reshape_115:out0';assert np.count_nonzero(data[fc]['bias'])==0
  names={k:f'row_{k}_{7000+b*10+i}' for i,k in enumerate(['abs','max','scale','normalized','restored'])}
  def add(k,op,inputs,params=None):
   n=names[k];assert n not in g['Layers'];g['Layers'][n]={'name':n,'op':op,'parameters':params or {},'inputs':inputs,'outputs':['out0']};q['quantize_parameters']['@'+n+':out0']={'qtype':'float16','quantizer':'float16'};added.append(n)
  add('abs','abs',[source]);add('max','reducemax',['@'+names['abs']+':out0'],{'axis_list':[1],'keep_dims':True});add('scale','clipbyvalue',['@'+names['max']+':out0'],{'clip_value_min':2**-14,'clip_value_max':65504.0});add('normalized','divide',[source,'@'+names['scale']+':out0'])
  g['Layers'][converter]['inputs']=['@'+names['normalized']+':out0'];q['quantize_parameters']['@'+converter+':out0']={'qtype':'i16','quantizer':'dynamic_fixed_point','rounding':'rtne','max_value':1.0,'min_value':-1.0,'fl':14}
  q['quantize_parameters']['@'+fc+':out0']={'qtype':'float16','quantizer':'float16'}
  for key,node in list(g['Layers'].items()):
   old=node.get('inputs',[])
   if '@'+fc+':out0' in old:
    rewires.append({'node':key,'before':list(old)});node['inputs']=['@'+names['restored']+':out0' if v=='@'+fc+':out0' else v for v in old]
  add('restored','multiply',['@'+fc+':out0','@'+names['scale']+':out0'])
 restored=copy.deepcopy(g)
 for n in added:del restored['Layers'][n]
 for r in rewires:restored['Layers'][r['node']]['inputs']=r['before']
 for b in range(2):restored['Layers'][f'down_i16_{2000+b}']['inputs']=og['Layers'][f'down_i16_{2000+b}']['inputs']
 assert restored==og
 changed=set('@'+n+':out0' for n in added)|{f'@down_i16_{2000+b}:out0' for b in range(2)}|{f'@b{b}_fullconnect_116:out0' for b in range(2)}
 assert all(q['quantize_parameters'][k]==v for k,v in oq['quantize_parameters'].items() if k not in changed)
 write(D/'decoder.json',g);(D/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(D/'decoder.data').symlink_to((S/'decoder.data').resolve());shutil.copy2(S/'tensor-inputmeta.yml',D/'tensor-inputmeta.yml')
 write(D/'source-audit.json',{'source_sha256':frozen,'added_nodes':added,'rewired':rewires,'weight_source_identical':True,'zero_bias_asserted':True,'only_local_normalize_and_restore_paths':True,'normalization':'per row max(abs(x)),clamp min2^-14 max65504; FP16 x/scale; I16/fl14; original I16 weights; FP16 FCoutput * scale','offline_evidence_sha256':sha(G/'real-down-row-scale/ANALYSIS.json'),'calibration':'none','not_approved':True})
 B=G/'cases/c512-m32-l2';cmd=[x.replace(str(B),str(D)) for x in json.loads((B/'conversion.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-row-scale-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=sdk;st.update(state='compiling',command=cmd);save()
 with (D/'compile.log').open('w') as log:
  try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
 assert rc==0,rc
 assert all(sha(S/n)==h for n,h in frozen.items())
 nb=D/'export_nbg_unify/network_binary.nb';st.update(state='complete',ended=time.time(),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size);save()
except BaseException:st.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
