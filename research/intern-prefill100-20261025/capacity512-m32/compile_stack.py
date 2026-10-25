"""Compile a C512 M32 stack from frozen C1280 M16 original/I16 prefill grids. No board access."""
from pathlib import Path
import json,copy,yaml,hashlib,subprocess,time,fcntl,traceback,shutil,os
D=Path(__file__).resolve().parent;R=D.parent;BASE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export');T8=Path('/opt/opi-lab/intern-nextopt-20261025/prefill8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state={'status':'preparing','started':time.time(),'pid':os.getpid(),'scope':'capacity1280 to512 and prefill16 to32; same source weights/quantizers/payload; original L2 first-chunk graph included; not board qualified','cases':[]}
def save():(D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 save();lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert Path('/proc/539/cmdline').read_bytes()==b'sleep\x0021600\x00'
 mtemplate=json.loads((R.parent/'intern-context256-20261025/prefill8-stack256/plan.json').read_text())['M_changes_from_c256_t16'];assert len(mtemplate)==41
 for kind,tokens in [('prefill32',32),('original32',32)]:
  for first in (range(0,24,2) if kind=='prefill32' else [2]):
   assert shutil.disk_usage('/mnt/f').free>8*1024**3
   src=(R/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2') if kind=='prefill32' and first==2 else BASE/'w8-fused2-c1280-t16-cap1-productwide'/f'l{first}'
   out=D/'cases'/kind/f'l{first}';out.mkdir(parents=True,exist_ok=False)
   g=json.loads((src/'decoder.json').read_text());changes=[]
   for i in range(2):
    targets=[(f'b{i}_mask_104','shape',3,1280+16,512+tokens),(f'b{i}_past_k_102','shape',2,1280,512),(f'b{i}_past_v_103','shape',2,1280,512)]
    newuid=11007 if tokens==1 else 11005;pastuid=11006 if tokens==1 else 11004
    targets += [(f'b{i}_split_probs_new_{newuid}','begin',3,1280,512),(f'b{i}_split_probs_past_{pastuid}','size',3,1280,512)]
    for node,field,index,before,after in targets:
     arr=g['Layers'][node]['parameters'][field];assert arr[index]==before,(kind,first,node,arr);arr[index]=after;changes.append({'node':node,'field':field,'index':index,'before':before,'after':after})
   for ch in mtemplate:
    if 'mask_104/parameters/shape/3' in ch['path']:continue
    parts=ch['path'].strip('/').split('/');node=g
    for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
    key=int(parts[-1]) if isinstance(node,list) else parts[-1]
    before=ch['before'];assert before in [16,112] and node[key]==before,(ch['path'],node[key],before)
    after=tokens if before==16 else 7*tokens;node[key]=after;changes.append({'path':ch['path'],'before':before,'after':after})
   assert len(changes)==49
   hashes={n:sha(src/n) for n in ['decoder.json','decoder.quantize','decoder.data','export_nbg_unify/network_binary.nb']}
   (out/'decoder.json').write_text(json.dumps(g,indent=2)+'\n');(out/'decoder.data').symlink_to((src/'decoder.data').resolve());shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
   meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
   for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
   (out/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
   conv=json.loads((src/'conversion.json').read_text());conv.update(capacity=512,original_capacity=1280,query_tokens=tokens,first_layer=first,fusion_width=2,source_graph=str(src),scope=state['scope'],capacity_changes=changes)
   (out/'conversion.json').write_text(json.dumps(conv,indent=2)+'\n')
   row={'kind':kind,'first_layer':first,'query_tokens':tokens,'source':str(src),'source_hashes':hashes,'output':str(out),'changes':changes};state['cases'].append(row);state.update(status='compiling',current=f'{kind}-l{first}');save()
   name=f'opi-c512-m32-{kind}-{first}'
   cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(R)+':'+str(R),'-v',str(BASE)+':'+str(BASE)+':ro','-v',str(T8)+':'+str(T8)+':ro','-w',str(out),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
   row['command']=cmd;start=time.monotonic();save()
   with (out/'export.log').open('w') as log:
    try:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','-t','2',name],stdout=log,stderr=subprocess.STDOUT,timeout=20);raise
   row.update(returncode=p.returncode,seconds=time.monotonic()-start);assert p.returncode==0
   nb=out/'export_nbg_unify/network_binary.nb';assert nb.is_file();row.update(nb_sha256=sha(nb),nb_bytes=nb.stat().st_size)
   assert sha(out/'decoder.quantize')==hashes['decoder.quantize'] and sha(out/'decoder.data')==hashes['decoder.data']
   for n,h in hashes.items():assert sha(src/n)==h
   payload=out/'export/decoder.export.data';prior=src/'export/decoder.export.data';assert sha(payload)==sha(prior),'capacity edit changed constant payload'
   row.update(payload_exact_source=True,payload_sha256=sha(payload),quantizer_exact_source=True)
   payload.unlink();payload.symlink_to(prior.resolve());row['status']='compiled_pending_actual_port_audit';save();print(kind,first,'compiled',flush=True)
 state.update(status='complete_pending_actual_ports_and_board',ended=time.time(),binary_bytes=sum(x['nb_bytes'] for x in state['cases']));save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
