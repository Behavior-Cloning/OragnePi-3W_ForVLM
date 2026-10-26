from pathlib import Path
import json,yaml,shutil,subprocess,time,fcntl,hashlib
P=Path(__file__).resolve().parent;S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vitfull-20261027-r1');base=S/'compiled13'
g=json.loads((base/'decoder.json').read_text())['Layers']
def ancestors(root,stops):
 seen=set()
 def walk(n):
  if n in stops or n in seen:return
  assert g[n]['op'] not in ['input','fullconnect'],n
  seen.add(n)
  for v in g[n].get('inputs',[]):
   if v.startswith('@'):walk(v[1:].split(':')[0])
 walk(root);return seen
norm=ancestors('Mul_6_output_0_41',{'Add_6_output_0_2'})|ancestors('Mul_1_output_0_69',{'hidden_85'})
gelu=ancestors('Mul_13_output_0_13',{'reshape_102'})-{'Mul_13_output_0_13'}
records=[]
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for tag,nodes in [('norm',norm),('gelu',gelu),('norm-gelu',norm|gelu)]:
  out=S/('selective-'+tag+'-l13');out.mkdir(exist_ok=False)
  for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:shutil.copy2(base/n,out/n)
  q=yaml.safe_load((base/'decoder.quantize').read_text())
  for n in nodes:q['quantize_parameters']['@'+n+':out0']={'qtype':'float16','quantizer':'float16'}
  (out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
  cmd=['docker','run','--rm','--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(S)+':/work','-w','/work/'+out.name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
  with (P/('selective-'+tag+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=240)
  nb=out/'export_nbg_unify/network_binary.nb';row={'tag':tag,'floating_nodes':sorted(nodes),'returncode':z.returncode,'command':cmd}
  if z.returncode==0:
   assert nb.exists();shutil.copy2(nb,P/('selective-'+tag+'.nb'));row['sha256']=hashlib.sha256(nb.read_bytes()).hexdigest()
  records.append(row);(P/'selective-compile-status.json').write_text(json.dumps(records,indent=2)+'\n');print(tag,z.returncode,flush=True)
