from pathlib import Path
import subprocess,sys,json,time
r=Path(__file__).resolve().parent;records=[]
for kind,layers in [('decode',range(0,24,2)),('head',range(24,28)),('prefill',range(0,24,2))]:
 for layer in layers:
  out=Path('/opt/opi-lab/intern1b-int4-20261021')/kind/('l'+str(layer));m=out/'manifest.json'
  if m.exists() and json.loads(m.read_text()).get('exit')==0:records.append(json.loads(m.read_text()));continue
  print('START',kind,layer,flush=True)
  with (r/('compile-'+kind+'-l'+str(layer)+'.log')).open('w') as log:z=subprocess.run([sys.executable,str(r/'compile_candidate.py'),'--kind',kind,'--layer',str(layer)],stdout=log,stderr=subprocess.STDOUT)
  if m.exists():records.append(json.loads(m.read_text()))
  (r/'compile-progress.json').write_text(json.dumps({'status':'running' if z.returncode==0 else 'failed','records':records},indent=2))
  if z.returncode:raise RuntimeError('Compilation failed '+kind+str(layer))
  print('DONE',kind,layer,records[-1].get('binary_bytes'),flush=True)
(r/'compile-progress.json').write_text(json.dumps({'status':'complete','records':records},indent=2));print('ALL COMPILED',flush=True)
