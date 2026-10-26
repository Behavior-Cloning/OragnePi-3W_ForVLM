from pathlib import Path
import sys,json,hashlib,tarfile,numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit-endpoints-r1';out=P/'endpoint-results';out.mkdir(exist_ok=False);assets=P/'endpoint-assets';assets.mkdir(exist_ok=False)
cases=json.loads((P/'endpoints-status.json').read_text())['cases'];files={'probe':P/'endpoint-probe'}
for c in cases:
 f=P/'endpoints-nbg'/(c['name']+'.nb');assert hashlib.sha256(f.read_bytes()).hexdigest()==c['nb_sha256'];files[c['name']+'.nb']=f
 for i in [1,2]:
  f=assets/(c['name']+str(i)+'.f32');np.load(Path(c['source'])/('input%d.npy'%i)).astype('float32').tofile(f);files[f.name]=f
with b.hostlock():
 locked('mkdir '+remote+' && mkdir '+remote+'/results',check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/probe',check=True)
 for name,ni,no in [('stem',150528,263168),('projector',263168,57344)]:
  for i in [1,2]:
   n=name+str(i);z=locked('cd '+remote+' && taskset -c 7 ./probe '+name+'.nb '+n+'.f32 results/'+n+'.f32 results/'+n+'.json '+str(ni)+' '+str(no)+' > results/'+n+'.log 2>&1',capture_output=True,timeout=90);assert z.returncode==0,(name,i,z.returncode)
 data=locked('cd '+remote+' && tar czf - results',capture_output=True,check=True,timeout=90).stdout
(out/'raw.tar.gz').write_bytes(data)
with tarfile.open(out/'raw.tar.gz') as t:t.extractall(out,filter='data')
print('complete')
