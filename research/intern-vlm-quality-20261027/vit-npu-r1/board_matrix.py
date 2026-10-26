from pathlib import Path
import subprocess,os,json,hashlib,time
P=Path(__file__).resolve().parent;out=P/'matrix-results';out.mkdir(exist_ok=False);cfg=json.loads((P/'matrix-config.json').read_text());records=[]
for name,h in cfg['sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
sequence=[(112,'int16')]+[(s,d) for s in [112,224] for d in ['fp16','int8','int16']]
try:
 for index,(side,dtype) in enumerate(sequence):
  for image in [1,2]:
   name='%02d-s%d-%s-i%d'%(index,side,dtype,image);rows=(side//14)**2+1
   env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')}
   with (out/(name+'.log')).open('w') as log:
    z=subprocess.run(['taskset','-c','7',str(P/'vit-matrix-probe'),str(P/('s%d-%s.nb'%(side,dtype))),str(P/('s%d-input%d.f32'%(side,image))),str(out/(name+'.f32')),str(out/(name+'.json')),str(rows)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
   assert z.returncode==0,(name,z.returncode)
   m=json.loads((out/(name+'.json')).read_text());assert m['released_bytes']==0 and m['differing_elements']==0
   records.append({'name':name,'side':side,'dtype':dtype,'image':image,'metrics':m,'output_sha256':hashlib.sha256((out/(name+'.f32')).read_bytes()).hexdigest()})
   (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 result={'status':'complete','scope':'first ViT block only; repeated development-image inputs; not full tower or VQA quality','records':records}
except BaseException:
 import traceback
 result={'status':'failed','error':traceback.format_exc(),'records':records};raise
finally:(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
