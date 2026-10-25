"""Bounded A8 gate/up experiment on current DFP weights and M64 prefill.
Historical M1/PC-weight A8 failed; this is a separate throughput/precision screen.
No activation calibration or semantic evaluation is used to choose grids.
"""
from pathlib import Path
import copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid,yaml,numpy as np
G=Path(__file__).resolve().parent;T=G.parent;R=G/'compact-gateup-a8-m64';R.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
state={'state':'waiting_sdk_lock','pid':os.getpid(),'started':time.time(),'cases':[]}
def save():write(R/'status.json',state)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
base=G/'compact-past128-screen/m64-l0';SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
try:
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
 sys.path.insert(0,str(T.parents[1]/'npu/kv-cache'));from fc_input_precision import convert_fc_inputs;from fused_contract import verify_contract
 frozen={n:sha(base/n) for n in ['decoder.json','decoder.data','decoder.quantize','tensor-inputmeta.yml','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
 meta=json.loads((base/'conversion.json').read_text());assert meta['capacity']==128 and meta['query_tokens']==64 and meta['first_layer']==0
 helper=T.parents[1]/'npu/kv-cache/fc_input_precision.py'
 for rangebits in [0,1]:
  variant='gateup-a8-range'+str(rangebits);dst=R/variant;dst.mkdir();g=json.loads((base/'decoder.json').read_text());q=yaml.safe_load((base/'decoder.quantize').read_text());data=np.load(base/'decoder.data',allow_pickle=True).item();oldg=copy.deepcopy(g);oldq=copy.deepcopy(q)
  change=convert_fc_inputs(g,q,data,896,'gateup',rangebits);assert len(change['changes'])==4
  assert all(oldq['quantize_parameters']['@'+x['linear']+':weight']['quantizer']=='dynamic_fixed_point' for x in change['changes'])
  restored=copy.deepcopy(g)
  for x in change['changes']:restored['Layers'][x['linear']]['inputs']=[x['source']]
  for name in set(g['Layers'])-set(oldg['Layers']):del restored['Layers'][name]
  assert restored==oldg
  newkeys=set(q['quantize_parameters'])-set(oldq['quantize_parameters']);assert len(newkeys)==4
  assert all(q['quantize_parameters'][k]==v for k,v in oldq['quantize_parameters'].items())
  write(dst/'decoder.json',g);(dst/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(dst/'decoder.data').symlink_to((base/'decoder.data').resolve());shutil.copy2(base/'tensor-inputmeta.yml',dst/'tensor-inputmeta.yml')
  write(dst/'source-audit.json',{'source_sha256':frozen,'helper_sha256':sha(helper),'change':change,'only_four_local_converter_paths':True,'all_original_quantizers_unchanged':True,'weight_source_identical':True,'calibration':'none; existing range shifted by8 bits, range1 halves range explicitly','no_quality_approval':True})
  cmd=[x.replace(str(base),str(dst)) for x in meta['command']];cmd[cmd.index('--name')+1]='opi-gateup-a8-'+uuid.uuid4().hex[:8]
  state.update(state='compiling',current=variant);save();started=time.monotonic()
  with (dst/'export.log').open('w') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
   except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
  row={'variant':variant,'returncode':rc,'seconds':time.monotonic()-started,'source_sha256':frozen,'command':cmd}
  if rc==0:
   ports=verify_contract(dst,2);assert ports==meta['ports'],(ports,meta['ports'])
   payload=dst/'export/decoder.export.data';row['payload_equal']=sha(payload)==frozen['export/decoder.export.data']
   if row['payload_equal']:payload.unlink();payload.symlink_to((base/'export/decoder.export.data').resolve())
   nb=dst/'export_nbg_unify/network_binary.nb';row.update(nb_sha256=sha(nb),nb_bytes=nb.stat().st_size,port_order_verified=True,ports=ports)
  assert all(sha(base/n)==h for n,h in frozen.items())
  write(dst/'conversion.json',row);state['cases'].append(row);save();print(variant,rc,row.get('nb_sha256'),flush=True)
 state.update(state='complete',ended=time.time());save()
except BaseException:state.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
