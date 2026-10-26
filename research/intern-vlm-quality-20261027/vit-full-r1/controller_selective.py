from pathlib import Path
import sys,json,hashlib,tarfile,time
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
records=json.loads((P/'selective-compile-status.json').read_text());assert len(records)==3
remote='/tmp/opi-vit-selective-r1';out=P/'selective-results';assert not out.exists();out.mkdir();cases=[]
with b.hostlock():
 locked('mkdir '+remote+' && mkdir '+remote+'/results',check=True)
 for row in records:
  if row['returncode']:continue
  f=P/('selective-'+row['tag']+'.nb');assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];upload(f,remote+'/'+row['tag']+'.nb')
  for image in [1,2]:
   name=row['tag']+'-i'+str(image);cmd='cd '+remote+' && taskset -c 7 /tmp/opi-vit-isolate-r1/probe '+row['tag']+'.nb /tmp/opi-vit-isolate-r1/l13-i'+str(image)+'-reference.f32 results/'+name+'.f32 results/'+name+'.json 257 > results/'+name+'.log 2>&1'
   z=locked(cmd,capture_output=True,timeout=90);assert z.returncode==0,(name,z.returncode)
   cases.append({'name':name,'image':image,'tag':row['tag']});(P/'selective-board-status.json').write_text(json.dumps({'status':'running','cases':cases}))
 data=locked('cd '+remote+' && tar czf - results',check=True,capture_output=True,timeout=90).stdout
(P/'selective-results.tar.gz').write_bytes(data)
with tarfile.open(P/'selective-results.tar.gz') as t:t.extractall(out,filter='data')
(P/'selective-board-status.json').write_text(json.dumps({'status':'complete','cases':cases},indent=2)+'\n');print('completed',len(cases))
