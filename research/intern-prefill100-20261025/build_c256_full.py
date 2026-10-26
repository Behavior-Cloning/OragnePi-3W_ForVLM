#!/usr/bin/env python3
import json,hashlib,shutil,subprocess,time,traceback,os,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent
SRC=R/'capacity512-m32/cases/prefill32'
OUT=R/'capacity256-m32/cases/prefill32'
OUT.mkdir(parents=True,exist_ok=True)
state={'status':'preparing','started':time.time(),'capacity':256,'source_capacity':512,'query_tokens':32,'layers':list(range(0,24,2)),'cases':[]}
def save(): (R/'capacity256-m32/status.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def setpath(root,path,val):
 parts=path.strip('/').split('/');node=root
 for part in parts[:-1]: node=node[int(part)] if isinstance(node,list) else node[part]
 key=int(parts[-1]) if isinstance(node,list) else parts[-1]; old=node[key];node[key]=val;return old
try:
 lock=open('/tmp/opi-prefill-c256-build.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (R/'capacity256-m32/README.md').write_text('# C256 M32 plain 全层图构建\n\n从已验证 C512 M32 plain 图只改变 KV capacity 512→256；query32、权重、quantizer、Transformer 结构保持不变。构建成功不等于板端完整性能达标。\n')
 save()
 for first in state['layers']:
  src=SRC/f'l{first}'; out=OUT/f'l{first}'; out.mkdir(exist_ok=True)
  g=json.loads((src/'decoder.json').read_text()); changes=[]
  for i in range(2):
   edits=[(f'b{i}_mask_104','shape',3,544,288),(f'b{i}_past_k_102','shape',2,512,256),(f'b{i}_past_v_103','shape',2,512,256),(f'b{i}_split_probs_new_11005','begin',3,512,256),(f'b{i}_split_probs_past_11004','size',3,512,256)]
   for node,field,index,before,after in edits:
    arr=g['Layers'][node]['parameters'][field]; assert arr[index]==before,(first,node,arr)
    arr[index]=after; changes.append({'node':node,'field':field,'index':index,'before':before,'after':after})
  (out/'decoder.json').write_text(json.dumps(g,indent=2)+'\n')
  (out/'decoder.data').symlink_to((src/'decoder.data').resolve())
  shutil.copy2(src/'decoder.quantize',out/'decoder.quantize')
  meta=__import__('yaml').safe_load((src/'tensor-inputmeta.yml').read_text())
  for port in meta['input_meta']['databases'][0]['ports']:
   node=g['Layers'][port['lid']]; port['shape']=node['parameters']['shape']
  (out/'tensor-inputmeta.yml').write_text(__import__('yaml').safe_dump(meta,sort_keys=False))
  row={'layer':first,'source':str(src),'changes':changes,'source_hashes':{n:sha(src/n) for n in ('decoder.json','decoder.data','decoder.quantize')}}
  cmd=['docker','run','--rm','--name',f'opi-c256-m32-{first}','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(R.parent)+':'+str(R.parent),'-v','/mnt/f/Files/OPi-Lab:/mnt/f/Files/OPi-Lab:ro','-v','/opt/opi-lab:/opt/opi-lab:ro','-w',str(out),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
  row['command']=cmd;state['status']='compiling';state['current']=first;state['cases'].append(row);save()
  with (out/'export.log').open('w') as log:
   rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
  row['returncode']=rc
  assert rc==0
  nb=out/'export_nbg_unify/network_binary.nb'; assert nb.is_file() and nb.stat().st_size>0
  row['nb_sha256']=sha(nb);row['nb_bytes']=nb.stat().st_size
  row['decoder_data_sha256']=sha(out/'decoder.data');row['decoder_quantize_sha256']=sha(out/'decoder.quantize')
  assert row['decoder_data_sha256']==row['source_hashes']['decoder.data']
  assert row['decoder_quantize_sha256']==row['source_hashes']['decoder.quantize']
  row['status']='compiled';save()
 state.update(status='complete',ended=time.time());save()
except BaseException:
 state.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
