from pathlib import Path
import copy,json,yaml,numpy as np,hashlib,subprocess,sys,os,time,traceback,uuid,fcntl,shutil
D=Path(__file__).resolve().parent;T=D.parent;R=T.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def signature(x):return {k:v for k,v in x.items() if k not in ['min_value','max_value']}
s={'status':'waiting_compile_lock','pid':os.getpid(),'started':time.time(),'cases':[]}
def save():write(D/'status.json',s)
save()
try:
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);sys.path.insert(0,str(R/'npu/kv-cache'));from fused_contract import verify_contract
  for shape,S,rows,past in [('m32-p448',T/'down-outlier-split-r2/full-c448-m32',32,448),('m64-p128',T/'compact-m64-r4/graphs/l2',64,128)]:
   for mode in ['scores','value','both']:
    name=shape+'-'+mode;P=D/name;P.mkdir(exist_ok=False);g=json.loads((S/'decoder.json').read_text());og=copy.deepcopy(g);q=yaml.safe_load((S/'decoder.quantize').read_text());oq=copy.deepcopy(q);L=g['Layers'];Q=q['quantize_parameters'];added=[];modified=[]
    frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
    def ref(k):return '@'+k+':out0'
    def add(k,op,inputs,params,qr):
     assert k not in L;L[k]={'name':k,'op':op,'inputs':inputs,'parameters':params,'outputs':['out0']};Q[ref(k)]=copy.deepcopy(Q[qr]);added.append(k);return ref(k)
    for b in range(2):
     pr=f'b{b}_';pk=ref(pr+'past_k_102');nk=ref(pr+'new_k_4');pv=ref(pr+'past_v_103');nv=ref(pr+'new_v_3');score=pr+'MatMul_3_output_0_66';value=pr+'MatMul_4_output_0_41';soft=ref(pr+'Softmax_output_0_50')
     assert L[score]['op']=='concat' and L[value]['op']=='add'
     assert signature(Q[pk])==signature(Q[nk]) and signature(Q[pv])==signature(Q[nv])
     if mode in ['scores','both']:
      sp=pr+'split_scores_past_11001';sn=pr+'split_scores_new_11003';assert signature(Q[ref(sp)])==signature(Q[ref(sn)])==signature(Q[ref(score)])
      query=L[sp]['inputs'][0];assert query==L[sn]['inputs'][0]
      joint=add(pr+'joint_k_'+str(13000+b*10),'concat',[pk,nk],{'dim':2},pk);kt=add(pr+'joint_kt_'+str(13001+b*10),'permute',[joint],{'perm':[0,1,3,2]},pk)
      L[score].update(op='matmul',inputs=[query,kt],parameters={'transpose_a':False,'transpose_b':False});modified.append(score)
     if mode in ['value','both']:
      joint=add(pr+'joint_v_'+str(13002+b*10),'concat',[pv,nv],{'dim':2},pv)
      L[value].update(op='matmul',inputs=[soft,joint],parameters={'transpose_a':False,'transpose_b':False});modified.append(value)
    live=set()
    def visit(k):
     if k in live:return
     live.add(k)
     for r in L[k].get('inputs',[]):visit(r[1:].rsplit(':',1)[0])
    for k,v in list(L.items()):
     if v['op']=='output':visit(k)
    removed=sorted(set(L)-live);data=np.load(S/'decoder.data',allow_pickle=True).item();assert not set(removed).intersection(data)
    for k in removed:
     assert L[k]['op'] in ['matmul','slice','permute'];del L[k];del Q[ref(k)]
    assert all(L[k]==v for k,v in og['Layers'].items() if k not in removed+modified)
    assert all(Q[k]==v for k,v in oq['quantize_parameters'].items() if k not in [ref(n) for n in removed])
    assert all(k.rsplit('_',1)[-1].isdigit() for k in L)
    write(P/'decoder.json',g);(P/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(P/'decoder.data').symlink_to((S/'decoder.data').resolve());shutil.copy2(S/'tensor-inputmeta.yml',P/'tensor-inputmeta.yml')
    base=T/'compact-m64-r4/graphs/l0';cmd=json.loads((base/'build.json').read_text())['compile_command'];cmd[cmd.index('-w')+1]=str(P);cmd[cmd.index('--name')+1]='opi-joint-r10-'+uuid.uuid4().hex[:8]
    s.update(status='compiling',current=name);save()
    with (P/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=360)
    assert z.returncode==0,z.returncode;assert all(sha(S/n)==h for n,h in frozen.items());ports=verify_contract(P,2);oldports=verify_contract(S,2);assert ports['input_order']==oldports['input_order'] and ports['output_order']==oldports['output_order']
    nb=P/'export_nbg_unify/network_binary.nb';row={'name':name,'shape':shape,'mode':mode,'parent':str(S),'source_sha256':frozen,'query_rows':rows,'past':past,'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'ports':ports,'data_unchanged':sha(P/'decoder.data')==frozen['decoder.data'],'export_payload_equal':sha(P/'export/decoder.export.data')==frozen['export/decoder.export.data'],'existing_quantizers_preserved':True,'reduction_regrouped':mode in ['value','both'],'added':added,'removed':removed,'modified':modified,'original_matmuls':sum(v['op']=='matmul' for v in og['Layers'].values()),'candidate_matmuls':sum(v['op']=='matmul' for v in L.values()),'command':cmd,'status':'compiled-unvalidated'}
    write(P/'build.json',row);s['cases'].append(row);save();print(name,row['nb_sha256'],flush=True)
  s.update(status='compiled-unvalidated',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
