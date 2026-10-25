from pathlib import Path
import json,yaml,copy,subprocess,fcntl,time,hashlib,traceback,os
D=Path(__file__).resolve().parent;T=D.parent.parent
B=D.parent/'down-minimal-v2';C=D.parent/'candidates-down-i16-conv'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(): (D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'status':'preparing','pid':os.getpid(),'started':time.time(),'scope':'SDK isolated mixed I16 input/weight to FP16 output support test; no board, speed or quality claims','cases':[]};save()
try:
 lock=Path('/tmp/opi-prefill-compile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for op in ['fc','conv']:
  S=B/(op+'-i16-fl16')
  assert json.loads((S/'result.json').read_text())['returncode']==0
  for mode in ['i16-control','fp16-output']:
   name=op+'-'+mode;O=D/name;O.mkdir()
   for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:
    (O/n).write_bytes((S/n).read_bytes())
   q=yaml.safe_load((S/'decoder.quantize').read_text());original=copy.deepcopy(q)
   if mode=='fp16-output':
    for key in ['@project_2:out0','@output_0:out0']:q['quantize_parameters'][key]={'qtype':'float16','quantizer':'float16'}
   assert q['quantize_parameters']['@input_1:out0']==original['quantize_parameters']['@input_1:out0']
   assert q['quantize_parameters']['@project_2:weight']==original['quantize_parameters']['@project_2:weight']
   (O/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
   cmd=json.loads((C/'status.json').read_text())['command'];old=str(C/'c512-m32-l2');cmd=[x.replace(old,str(O)) for x in cmd];cmd[cmd.index('--name')+1]='opi-down-outfp16-'+name
   state.update(status='compiling',current=name);save();start=time.monotonic()
   with (O/'export.log').open('wb') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=180)
   nb=O/'export_nbg_unify/network_binary.nb';lines=(O/'export.log').read_text(errors='replace').splitlines()
   entry={'case':name,'returncode':r.returncode,'seconds':time.monotonic()-start,'nb_sha256':sha(nb) if nb.exists() else None,'nb_bytes':nb.stat().st_size if nb.exists() else None,'errors':[x for x in lines if any(z in x for z in ['CHECK STATUS','Unsupported','not support'])],'graph_sha256':sha(O/'decoder.json'),'quant_sha256':sha(O/'decoder.quantize'),'data_sha256':sha(O/'decoder.data'),'source_data_sha256':sha(S/'decoder.data'),'quantize_parameters':q['quantize_parameters']}
   assert entry['data_sha256']==entry['source_data_sha256']
   (O/'result.json').write_text(json.dumps(entry,indent=2)+'\n');state['cases'].append(entry);save();print(name,r.returncode,flush=True)
   if mode=='i16-control' and (r.returncode!=0 or not nb.exists()):raise RuntimeError('I16 control failed; do not infer mixed type support')
 state.update(status='complete',ended=time.time());save()
except BaseException:state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
