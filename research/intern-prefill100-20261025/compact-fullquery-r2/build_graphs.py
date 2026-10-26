from pathlib import Path
import copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid,yaml
R=Path(__file__).resolve().parent;T=R.parent;G=T/'graphs';D=R/'graphs';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s={'status':'waiting_compile_lock','pid':os.getpid(),'started':time.time(),'cases':[]}
def save():write(R/'compile-status.json',s)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
 assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
 sys.path.insert(0,str(T.parents[1]/'npu/kv-cache'));from fused_contract import verify_contract
 b32=G/'cases/c512-m32-l0';b128=G/'cases/c512-m128-l0'
 g32=json.loads((b32/'decoder.json').read_text())['Layers'];g128=json.loads((b128/'decoder.json').read_text())['Layers']
 geom={k:(g32[k]['parameters'],g128[k]['parameters']) for k in g32 if g32[k].get('parameters')!=g128[k].get('parameters') and not k.endswith('_mask_104')}
 for layer in range(0,24,2):
  S=T/'down-outlier-split-r2/full-c448-m32' if layer==2 else T/'capacity448-m32/cases/prefill32'/('l'+str(layer))
  P=D/('l'+str(layer));P.mkdir();src={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
  g=json.loads((S/'decoder.json').read_text());old=copy.deepcopy(g);edits=[]
  for key,(a,b) in geom.items():
   for param in a:
    if a[param]==b[param]:continue
    assert isinstance(a[param],list) and len(a[param])==len(b[param])
    for axis,(before,after) in enumerate(zip(a[param],b[param])):
     if before==after:continue
     assert g['Layers'][key]['parameters'][param][axis]==before,(layer,key,param,axis)
     g['Layers'][key]['parameters'][param][axis]=after
   edits.append(key)
  for b in range(2):
   for kv in ['k','v']:
    key=f'b{b}_past_{kv}_{102 if kv=="k" else 103}';a=g['Layers'][key]['parameters']['shape'];assert a[2]==448;a[2]=128;edits.append(key)
   key=f'b{b}_mask_104';a=g['Layers'][key]['parameters']['shape'];assert a==[1,1,224,480];a[:]=[1,1,896,256];edits.append(key)
   key=f'b{b}_split_probs_past_11004';a=g['Layers'][key]['parameters']['size'];assert a[3]==448;a[3]=128
   key=f'b{b}_split_probs_new_11005';a=g['Layers'][key]['parameters']['begin'];assert a[3]==448;a[3]=128
  for key,x in g['Layers'].items():
   if key.startswith('outlier_slice_'):
    assert x['parameters']['size']==[32,1];x['parameters']['size']=[128,1];edits.append(key)
  # The transformation changes parameters only, never topology, weights or quantizers.
  undo=copy.deepcopy(g)
  for key,x in old['Layers'].items():undo['Layers'][key]['parameters']=x['parameters']
  assert undo==old
  write(P/'decoder.json',g);shutil.copy2(S/'decoder.quantize',P/'decoder.quantize');(P/'decoder.data').symlink_to((S/'decoder.data').resolve())
  meta=yaml.safe_load((S/'tensor-inputmeta.yml').read_text())
  for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
  (P/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
  basecmd=json.loads((b32/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(b32),str(P)) for x in basecmd]
  cmd[cmd.index('--name')+1]='opi-cq-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
  cmd[3:3]=['-v',str(T)+':'+str(T)]
  s.update(status='compiling',layer=layer);save();start=time.time()
  with (P/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  assert z.returncode==0,z.returncode
  assert all(sha(S/n)==h for n,h in src.items())
  assert sha(P/'decoder.quantize')==src['decoder.quantize']
  assert sha(P/'export/decoder.export.data')==src['export/decoder.export.data']
  ports=verify_contract(P,2);ref=verify_contract(S,2)
  assert ports['input_order']==ref['input_order'] and ports['output_order']==ref['output_order']
  nb=P/'export_nbg_unify/network_binary.nb';row={'first_layer':layer,'source':str(S),'source_sha256':src,'past_capacity':128,'query_tokens':128,'storage_capacity':448,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'compile_command':cmd,'ports':ports,'payload_equal':True,'quantizers_equal_to_parent':True,'seconds':time.time()-start,'status':'compiled-unvalidated'}
  write(P/'build.json',row);s['cases'].append(row);save();print(layer,row['nb_sha256'],flush=True)
 s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
