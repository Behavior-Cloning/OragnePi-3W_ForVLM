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
  for name,S,rows,past in [('m64-p128',T/'compact-m64-r4/graphs/l0',64,128),('m32-p448',T/'capacity448-m32/cases/prefill32/l0',32,448)]:
   P=R/name;P.mkdir(exist_ok=False);g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());oq=copy.deepcopy(q)
   frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']};edits=[]
   for b in range(2):
    key=f'@b{b}_Add_6_output_0_57:out0';source=f'@b{b}_Div_2_output_0_59:out0';assert q['quantize_parameters'][key]=={'qtype':'float16','quantizer':'float16'}
    assert q['quantize_parameters'][source]['qtype']=='i16'
    # Preserve the scale already used by unmasked attention scores; masked -10000 saturates.
    dst=copy.deepcopy(q['quantize_parameters'][source]);dst.pop('max_value',None);dst.pop('min_value',None)
    q['quantize_parameters'][key]=dst;edits.append({'key':key,'from':oq['quantize_parameters'][key],'to':dst})
   assert all(q['quantize_parameters'][k]==v for k,v in oq['quantize_parameters'].items() if k not in [e['key'] for e in edits])
   shutil.copy2(S/'decoder.json',P/'decoder.json');(P/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(P/'decoder.data').symlink_to((S/'decoder.data').resolve());shutil.copy2(S/'tensor-inputmeta.yml',P/'tensor-inputmeta.yml')
   base=T/'compact-m64-r4/graphs/l0';cmd=[x.replace(str(base),str(P)) for x in json.loads((base/'build.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-sm-i16-'+uuid.uuid4().hex[:8]
   s.update(status='compiling',current=name);save()
   with (P/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
   assert z.returncode==0,z.returncode
   assert all(sha(S/n)==h for n,h in frozen.items());assert sha(P/'export/decoder.export.data')==frozen['export/decoder.export.data']
   ports=verify_contract(P,2);oldports=verify_contract(S,2);assert ports['input_order']==oldports['input_order'] and ports['output_order']==oldports['output_order']
   nb=P/'export_nbg_unify/network_binary.nb';row={'name':name,'parent':str(S),'source_sha256':frozen,'query_rows':rows,'past':past,'quantizer_changes':edits,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'ports':ports,'payload_equal':True,'command':cmd,'status':'compiled-unvalidated','warning':'Precision change; masked value saturation must be assessed; not lossless'};write(P/'build.json',row);s['cases'].append(row);save();print(name,row['nb_sha256'],flush=True)
 s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
