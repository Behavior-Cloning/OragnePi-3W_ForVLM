from pathlib import Path
import subprocess,json,hashlib,os
R=Path(__file__).resolve().parent;P=R/'runtime';b=json.loads((R/'build-runtime.json').read_text());results=[]
cmd=['gcc','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-I/opt/opi-lab/restored-slim3-20261017/board-runtime/usr/include',str(P/'test_compact.c'),'-lm','-o',str(R/'test-compact-host')]
for tr in [0,1]:
 z=subprocess.run(cmd+['-DKEY_CACHE_TRANSPOSED='+str(tr)],capture_output=True,text=True);assert z.returncode==0,z.stderr
 for template,fp in [(False,False),(True,False),(True,True)]:
  env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')};env['ASAN_OPTIONS']='detect_leaks=1'
  if template:env['OPI_MASK_TEMPLATE']='1';env['OPI_MASK_TEMPLATE_VALIDATE']='1'
  if fp:env['OPI_TEST_MASK_FP16']='1'
  z=subprocess.run([str(R/'test-compact-host')],env=env,capture_output=True,text=True);results.append({'transpose':tr,'template':template,'fp16':fp,'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr})
  (R/'host-test.json').write_text(json.dumps(results,indent=2)+'\n');assert z.returncode==0,z.stderr
z=subprocess.run(b['command'],capture_output=True,text=True);assert z.returncode==0,z.stderr
b.update(binary_sha256=hashlib.sha256((P/'resident-compact').read_bytes()).hexdigest(),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.suffix in ['.c','.h']});(R/'build-runtime.json').write_text(json.dumps(b,indent=2)+'\n')
print('6 ASan/UBSan tests passed; binary',b['binary_sha256'])
