from pathlib import Path
import copy,json,yaml,hashlib,subprocess,sys,os,time,traceback,uuid,fcntl,shutil
R=Path(__file__).resolve().parent;T=R.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s={'status':'waiting_compile_lock','pid':os.getpid(),'started':time.time(),'cases':[]}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def save():write(R/'status.json',s)
save()
try:
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);sys.path.insert(0,str(T.parents[1]/'npu/kv-cache'));from fused_contract import verify_contract
  for name,S,rows,oldpast in [('l'+str(i),T/'compact-m64-r4/graphs'/('l'+str(i)),64,128) for i in range(0,24,2)]:
   if name in ['l0','l2']:
    P=T/'m64-p448-screen-r8'/name;row=json.loads((P/'build.json').read_text());assert row['parent']==str(S) and row['past']==448 and row['query_rows']==64 and sha(P/'export_nbg_unify/network_binary.nb')==row['nb_sha256'];assert all(sha(S/n)==h for n,h in row['source_sha256'].items())
    row.update(graph_dir=str(P),reused_existing_compilation=True);s['cases'].append(row);save();print(name,'reuse',row['nb_sha256'],flush=True);continue
   P=R/name;P.mkdir(exist_ok=False);g=json.loads((S/'decoder.json').read_text());og=copy.deepcopy(g)
   frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
   for b in range(2):
    for kv in ['k','v']:
     key=f'b{b}_past_{kv}_{102 if kv=="k" else 103}';a=g['Layers'][key]['parameters']['shape'];assert a[2]==oldpast;a[2]=448
    a=g['Layers'][f'b{b}_mask_104']['parameters']['shape'];assert a==[1,1,7*rows,oldpast+rows];a[3]=448+rows
    a=g['Layers'][f'b{b}_split_probs_past_11004']['parameters']['size'];assert a[3]==oldpast;a[3]=448
    a=g['Layers'][f'b{b}_split_probs_new_11005']['parameters']['begin'];assert a[3]==oldpast;a[3]=448
   undo=copy.deepcopy(g)
   for key,x in og['Layers'].items():
    if 'parameters' in x:undo['Layers'][key]['parameters']=x['parameters']
   assert undo==og
   write(P/'decoder.json',g);shutil.copy2(S/'decoder.quantize',P/'decoder.quantize');(P/'decoder.data').symlink_to((S/'decoder.data').resolve())
   meta=yaml.safe_load((S/'tensor-inputmeta.yml').read_text())
   for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
   (P/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
   base=T/'compact-m64-r4/graphs/l0';cmd=[x.replace(str(base),str(P)) for x in json.loads((base/'build.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-main64-r11-'+uuid.uuid4().hex[:8]
   s.update(status='compiling',current=name);save()
   with (P/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
   assert z.returncode==0,z.returncode
   assert all(sha(S/n)==h for n,h in frozen.items());assert sha(P/'export/decoder.export.data')==frozen['export/decoder.export.data'];assert sha(P/'decoder.quantize')==frozen['decoder.quantize']
   ports=verify_contract(P,2);oldports=verify_contract(S,2);assert ports['input_order']==oldports['input_order'] and ports['output_order']==oldports['output_order']
   nb=P/'export_nbg_unify/network_binary.nb';row={'name':name,'graph_dir':str(P),'reused_existing_compilation':False,'parent':str(S),'source_sha256':frozen,'query_rows':rows,'past':448,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'ports':ports,'payload_equal':True,'quantizers_unchanged':True,'command':cmd,'status':'compiled-unvalidated'};write(P/'build.json',row);s['cases'].append(row);save();print(name,row['nb_sha256'],flush=True)
 s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
