from pathlib import Path
import subprocess,concurrent.futures,json,shutil,time
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');rows=[]
def run(n):
 out=b/'w8element-imported'/n;out.mkdir(parents=True,exist_ok=True);base=b/'w8element-fp16/l0'
 with (r/'results'/('w8element-full-'+n+'.log')).open('w') as f:
  z=subprocess.run([str(b/'ref311/bin/python'),str(r/'scripts/export_w8element.py'),n],stdout=f,stderr=subprocess.STDOUT);assert z.returncode==0
  z=subprocess.run(['docker','run','--rm','--name','minicpm46-element-import-'+n,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-v',str(b)+':/work','-w','/work/w8element-imported/'+n,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','import','onnx','--model','/work/w8element-source/'+n+'/layer0.onnx','--output-model','decoder.json','--output-data','decoder.data'],stdout=f,stderr=subprocess.STDOUT,timeout=90);assert z.returncode==0
  assert json.loads((out/'decoder.json').read_text())==json.loads((base/'decoder.json').read_text())
  for fn in ['tensor-inputmeta.yml','decoder.quantize']:shutil.copyfile(base/fn,out/fn)
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8element.py'),n,'all'],stdout=f,stderr=subprocess.STDOUT)
  result=json.loads((b/'w8element-all'/n/'conversion.json').read_text());assert result['exit']==z.returncode
 return {'name':n,'exit':z.returncode}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for row in ex.map(run,['l'+str(i) for i in range(24) if i%4!=3 and i!=0]):
  rows.append(row);(r/'results/w8element-full-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
