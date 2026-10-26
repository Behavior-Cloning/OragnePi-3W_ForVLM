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
   P=R/name;P.mkdir(exist_ok=False);g=json.loads((S/'decoder.json').read_text());q=yaml.safe_load((S/'decoder.quantize').read_text());og=copy.deepcopy(g);oq=copy.deepcopy(q);added=[]
   frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
   for b in range(2):
    key=f'b{b}_Softmax_output_0_50';node=g['Layers'][key];assert node['op']=='softmax' and node['parameters']=={'sf_axis':-1,'beta':1.0};src=node['inputs'][0]
    before=f'b{b}_softmax_flatten_25000';sm=f'b{b}_softmax_flat_25001';oqt=q['quantize_parameters']['@'+key+':out0'];iqt=q['quantize_parameters'][src]
    g['Layers'][before]={'name':before,'op':'reshape','inputs':[src],'outputs':['out0'],'parameters':{'shape':[14*rows,past+rows]}}
    g['Layers'][sm]={'name':sm,'op':'softmax','inputs':['@'+before+':out0'],'outputs':['out0'],'parameters':copy.deepcopy(node['parameters'])}
    g['Layers'][key]={**node,'op':'reshape','inputs':['@'+sm+':out0'],'parameters':{'shape':[1,2,7*rows,past+rows]}}
    q['quantize_parameters']['@'+before+':out0']=copy.deepcopy(iqt);q['quantize_parameters']['@'+sm+':out0']=copy.deepcopy(oqt);added.extend([before,sm])
   undo=copy.deepcopy(g)
   for key in added:del undo['Layers'][key]
   for b in range(2):key=f'b{b}_Softmax_output_0_50';undo['Layers'][key]=og['Layers'][key]
   assert undo==og and all(q['quantize_parameters'][k]==v for k,v in oq['quantize_parameters'].items())
   write(P/'decoder.json',g);(P/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(P/'decoder.data').symlink_to((S/'decoder.data').resolve());shutil.copy2(S/'tensor-inputmeta.yml',P/'tensor-inputmeta.yml')
   base=T/'compact-m64-r4/graphs/l0';cmd=[x.replace(str(base),str(P)) for x in json.loads((base/'build.json').read_text())['compile_command']];cmd[cmd.index('--name')+1]='opi-sm-layout-'+uuid.uuid4().hex[:8]
   s.update(status='compiling',current=name);save()
   with (P/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
   assert z.returncode==0,z.returncode
   assert all(sha(S/n)==h for n,h in frozen.items());assert sha(P/'export/decoder.export.data')==frozen['export/decoder.export.data']
   ports=verify_contract(P,2);oldports=verify_contract(S,2);assert ports['input_order']==oldports['input_order'] and ports['output_order']==oldports['output_order']
   nb=P/'export_nbg_unify/network_binary.nb';row={'name':name,'parent':str(S),'source_sha256':frozen,'query_rows':rows,'past':past,'flat_shape':[14*rows,past+rows],'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'ports':ports,'all_original_quantizers_unchanged':True,'payload_equal':True,'command':cmd,'status':'compiled-unvalidated'};write(P/'build.json',row);s['cases'].append(row);save();print(name,row['nb_sha256'],flush=True)
 s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
