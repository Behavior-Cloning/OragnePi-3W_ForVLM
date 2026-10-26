#!/usr/bin/env python3
import json,hashlib,shutil,subprocess,time,traceback,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent; SRC=R/'capacity512-m32/cases/prefill32'; SRCD=R/'capacity512-range16/cases/decode'; OUT=R/'capacity448-m32'; OUT.mkdir(exist_ok=True)
state={'status':'preparing','started':time.time(),'capacity':448,'source_capacity':512,'layers':list(range(0,24,2)),'kinds':['decode','prefill32'],'cases':[]}
def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
try:
 lock=open('/tmp/opi-prefill-c448-build.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
 for kind in state['kinds']:
  (OUT/'cases'/kind).mkdir(parents=True,exist_ok=True)
  for first in state['layers']:
   src=(SRCD if kind=='decode' else SRC)/f'l{first}';out=OUT/'cases'/kind/f'l{first}';out.mkdir(exist_ok=True)
   g=json.loads((src/'decoder.json').read_text()); changes=[]
   if kind=='decode': edits=[]
   else: edits=[]
   for i in range(2):
    if kind=='decode':
     edits += [(f'b{i}_mask_104','shape',3,513,449),(f'b{i}_past_k_102','shape',2,512,448),(f'b{i}_past_v_103','shape',2,512,448),(f'b{i}_split_probs_new_11007','begin',3,512,448),(f'b{i}_split_probs_past_11006','size',3,512,448)]
    else:
     edits += [(f'b{i}_mask_104','shape',3,544,480),(f'b{i}_past_k_102','shape',2,512,448),(f'b{i}_past_v_103','shape',2,512,448),(f'b{i}_split_probs_new_11005','begin',3,512,448),(f'b{i}_split_probs_past_11004','size',3,512,448)]
   for node,field,index,before,after in edits:
    arr=g['Layers'][node]['parameters'][field];assert arr[index]==before,(kind,first,node,arr);arr[index]=after;changes.append({'node':node,'field':field,'index':index,'before':before,'after':after})
   (out/'decoder.json').write_text(json.dumps(g,indent=2)+'\n');(out/'decoder.data').symlink_to((src/'decoder.data').resolve());shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
   meta=__import__('yaml').safe_load((src/'tensor-inputmeta.yml').read_text())
   for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
   (out/'tensor-inputmeta.yml').write_text(__import__('yaml').safe_dump(meta,sort_keys=False))
   row={'kind':kind,'layer':first,'source':str(src),'changes':changes,'source_hashes':{n:sha(src/n) for n in ('decoder.json','decoder.data','decoder.quantize')}}
   cmd=['docker','run','--rm','--name',f'opi-c448-{kind}-{first}','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(R.parent)+':'+str(R.parent),'-v','/mnt/f/Files/OPi-Lab:/mnt/f/Files/OPi-Lab:ro','-v','/opt/opi-lab:/opt/opi-lab:ro','-w',str(out),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
   row['command']=cmd;state['status']='compiling';state['current']=f'{kind}-l{first}';state['cases'].append(row);save()
   with (out/'export.log').open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
   row['returncode']=rc;assert rc==0
   nb=out/'export_nbg_unify/network_binary.nb';assert nb.is_file() and nb.stat().st_size>0
   row['nb_sha256']=sha(nb);row['nb_bytes']=nb.stat().st_size;row['data_sha256']=sha(out/'decoder.data');row['quantize_sha256']=sha(out/'decoder.quantize');assert row['data_sha256']==row['source_hashes']['decoder.data'];assert row['quantize_sha256']==row['source_hashes']['decoder.quantize'];row['status']='compiled';save()
 state.update(status='complete',ended=time.time());save()
except BaseException:state.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
