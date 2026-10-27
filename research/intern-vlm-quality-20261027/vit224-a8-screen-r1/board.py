from pathlib import Path
import json,subprocess,hashlib,traceback
P=Path(__file__).resolve().parent;O=P/'results';O.mkdir(exist_ok=False);cfg=json.loads((P/'config.json').read_text());records=[]
for n,h in cfg['sha256'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h
try:
 for layer in [0,12]:
  for phase in ['base-before','input8','input-output8','base-after']:
   mode='base' if phase.startswith('base') else phase
   for im in [1,2]:
    name=f'l{layer:02d}-{phase}-i{im}'
    with (O/(name+'.log')).open('w') as log:z=subprocess.run(['taskset','-c','7',str(P/'probe'),str(P/f'l{layer:02d}-{mode}.nb'),str(P/f'l{layer:02d}-input{im}.f32'),str(O/(name+'.f32')),str(O/(name+'.json')),'257'],stdout=log,stderr=subprocess.STDOUT,timeout=90)
    assert z.returncode==0,(name,z.returncode)
    m=json.loads((O/(name+'.json')).read_text());assert m['released_bytes']==0 and m['differing_elements']==0
    records.append({'name':name,'layer':layer,'mode':mode,'phase':phase,'image':im,'metrics':m,'sha256':hashlib.sha256((O/(name+'.f32')).read_bytes()).hexdigest()})
 result={'status':'complete','scope':'2 selected blocks only; 2 images each; W8 fixed weights and FP16 sensitive operations retained; no whole-tower or VQA claim','records':records}
except BaseException:result={'status':'failed','error':traceback.format_exc(),'records':records};raise
finally:(O/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
