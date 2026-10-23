from pathlib import Path
import subprocess,yaml,numpy as np,shutil,json,concurrent.futures,threading,time
b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');r=Path(__file__).resolve().parents[1];results=[];lock=threading.Lock()
def run(name):
 src=b/'compiled-epssafe'/name;o=b/'compiled-w8'/name;o.mkdir(parents=True,exist_ok=True)
 for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:shutil.copyfile(src/n,o/n)
 q=yaml.safe_load((src/'decoder.quantize').read_text());data=np.load(src/'decoder.data',allow_pickle=True).item();changed=[]
 for key,d in data.items():
  if 'fullconnect' not in key or 'weight' not in d:continue
  w=d['weight'];scale=np.maximum(np.abs(w).max(axis=0)/127,np.finfo(np.float32).tiny)
  q['quantize_parameters']['@'+key+':weight']={'qtype':'i8','quantizer':'perchannel_symmetric_affine','rounding':'rtne','scale':scale.tolist(),'channel_dim':-1};changed.append({'key':key,'shape':list(w.shape)})
 assert changed
 (o/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));(o/'weight-change.json').write_text(json.dumps(changed,indent=2))
 cmd=['docker','run','--rm','--name','minicpm46-w8-'+name,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(b)+':/work','-w','/work/compiled-w8/'+name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
 start=time.time()
 with (o/'export.log').open('w') as f:
  try:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=240);code=z.returncode
  except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2','minicpm46-w8-'+name]);code=124
 row={'name':name,'exit':code,'seconds':time.time()-start,'changed_weights':len(changed)}
 with lock:results.append(row);(r/'results/w8-compile-status.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
 return row
names=['l0','head0','l3']+['l'+str(i) for i in range(24) if i not in [0,3]]+['head'+str(i) for i in range(1,8)]
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(run,names))
