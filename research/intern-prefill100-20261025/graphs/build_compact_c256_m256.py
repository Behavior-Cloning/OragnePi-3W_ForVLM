#!/usr/bin/env python3
from pathlib import Path
import copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid,yaml
G=Path(__file__).resolve().parent;T=G.parent;R=G/'compact-c256-m256';R.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
state={'state':'waiting_sdk_lock','pid':os.getpid(),'start':time.time(),'cases':[],'scope':'two-layer past128 query256, retain all effective history; not full runtime'}
def save():write(R/'status.json',state)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
try:
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
 base=G/'cases/c512-m256-l0';basecmd=json.loads((base/'conversion.json').read_text())['compile_command']
 sys.path.insert(0,str(T.parents[1]/'npu/kv-cache'));from fused_contract import verify_contract
 for variant in ['plain']:
  src=base;dst=R/variant;dst.mkdir()
  frozen={n:sha(src/n) for n in ['decoder.json','decoder.quantize','decoder.data','tensor-inputmeta.yml','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
  g=json.loads((src/'decoder.json').read_text());before=copy.deepcopy(g);edits=[]
  def change(lid,param,axis,old,new):
   x=g['Layers'][lid]['parameters'][param];assert x[axis]==old,(lid,x)
   x[axis]=new;edits.append({'layer':lid,'parameter':param,'axis':axis,'before':old,'after':new})
  for b in range(2):
   for k in ['k','v']:change(f'b{b}_past_{k}_{102 if k=="k" else 103}','shape',2,512,128)
   change(f'b{b}_mask_104','shape',3,768,512)
   change(f'b{b}_split_probs_past_11004','size',3,512,256)
   change(f'b{b}_split_probs_new_11005','begin',3,512,256)
   if variant=='row224':
    for j in range(8):change(f'b{b}_softmax_rowslice_{21000+2*j}','size',3,768,384)
  assert len(edits)==10
  restored=copy.deepcopy(g)
  for e in edits:restored['Layers'][e['layer']]['parameters'][e['parameter']][e['axis']]=e['before']
  assert restored==before
  write(dst/'decoder.json',g);shutil.copy2(src/'decoder.quantize',dst/'decoder.quantize');(dst/'decoder.data').symlink_to((src/'decoder.data').resolve())
  meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
  for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
  (dst/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False));write(dst/'graph-audit.json',{'variant':variant,'source_sha256':frozen,'edits':edits,'only_listed_geometry_changes':True,'past_capacity':256,'queries':256,'effective_history':256,'no_removed_active_history':True,'quantizers_byte_equal':True,'weight_source_same':True,'numeric_status':'not measured; retains all 256 active past positions'})
  cmd=[x.replace(str(base),str(dst)) for x in basecmd];cmd[cmd.index('--name')+1]='opi-compact-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1'];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
  state.update(state='compiling',current=variant);save();start=time.monotonic()
  with (dst/'export.log').open('w') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
   except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
  assert rc==0,rc
  assert all(sha(src/n)==h for n,h in frozen.items())
  payload=dst/'export/decoder.export.data';assert sha(payload)==frozen['export/decoder.export.data'];payload.unlink();payload.symlink_to((src/'export/decoder.export.data').resolve())
  ports=verify_contract(dst,2);expected=json.loads((base/'conversion.json').read_text());assert ports['input_order']==expected['input_order'] and ports['output_order']==expected['output_order']
  nb=dst/'export_nbg_unify/network_binary.nb';row={'variant':variant,'state':'compiled','capacity':128,'query_tokens':256,'first_layer':0,'fusion_width':2,'source_sha256':frozen,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'ports':ports,'port_order_verified_from_C':True,'payload_byte_equal':True,'quantize_byte_equal':sha(dst/'decoder.quantize')==frozen['decoder.quantize'],'command':cmd,'seconds':time.monotonic()-start};write(dst/'conversion.json',row);state['cases'].append(row);save();print(variant,row['nb_sha256'],flush=True)
 state.update(state='complete',end=time.time());save()
except BaseException:state.update(state='failed',error=traceback.format_exc(),end=time.time());save();raise
