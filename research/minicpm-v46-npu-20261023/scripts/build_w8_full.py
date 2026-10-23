from pathlib import Path
import subprocess,json,concurrent.futures,time,shutil,yaml,os
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');rows=[]
names=['l'+str(i) for i in range(24) if i not in [0,3]]+['head'+str(i) for i in range(1,8)]
def run(name):
 start=time.time();out=b/'w8-imported'/name;out.mkdir(parents=True,exist_ok=True);src=b/'w8retry-source'/name
 with (r/'results'/('w8full-'+name+'.log')).open('w') as f:
  z=subprocess.run([str(b/'ref311/bin/python'),str(r/'scripts/export_w8_parts.py'),name],stdout=f,stderr=subprocess.STDOUT)
  if z.returncode:return {'name':name,'stage':'onnx','exit':z.returncode}
  z=subprocess.run(['docker','run','--rm','--name','minicpm46-import-'+name,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-v',str(b)+':/work','-w','/work/w8-imported/'+name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','import','onnx','--model','/work/w8retry-source/'+name+'/layer0.onnx','--output-model','decoder.json','--output-data','decoder.data'],stdout=f,stderr=subprocess.STDOUT,timeout=90)
  if z.returncode:return {'name':name,'stage':'import','exit':z.returncode}
  baseline=b/'compiled-epssafe'/name
  assert json.loads((out/'decoder.json').read_text())==json.loads((baseline/'decoder.json').read_text()),'graph changed'
  shutil.copyfile(baseline/'tensor-inputmeta.yml',out/'tensor-inputmeta.yml')
  q=yaml.safe_load((baseline/'decoder.quantize').read_text());g=json.loads((out/'decoder.json').read_text())['Layers']
  for k,v in g.items():
   if v['op']=='fullconnect':q['quantize_parameters']['@'+k+':bias']={'qtype':'i64','quantizer':'dynamic_fixed_point','rounding':'rtne','fl':24}
  (out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8_household.py'),name,'all'],stdout=f,stderr=subprocess.STDOUT)
 return {'name':name,'stage':'export','exit':z.returncode,'seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for row in ex.map(run,names):
  rows.append(row);(r/'results/w8full-compile-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
print('COMPLETE',flush=True)
