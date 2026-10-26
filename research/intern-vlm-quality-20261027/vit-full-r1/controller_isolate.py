from pathlib import Path
import sys,json,hashlib,tarfile,numpy as np,time,shlex
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
O=P/'isolate-results';assert not O.exists();O.mkdir();A=P/'isolate-assets';A.mkdir(exist_ok=True)
remote='/tmp/opi-vit-isolate-r1';files={'probe':P.parent/'vit-npu-r1/vit-matrix-probe','w8.nb':P/'w8a16-l00.nb'};cases=[]
for layer in [0,12,13]:
 for image in [1,2]:
  for mode in (['reference'] if layer==0 else ['reference','propagated']):
   name=f'l{layer:02d}-i{image}-{mode}';f=A/(name+'.f32')
   if mode=='reference':np.load(P/f's224/l{layer:02d}-input{image}.npy').astype('float32').tofile(f)
   else:f.write_bytes((P/f'results/image{image}-layer{layer-1:02d}.f32').read_bytes())
   files[f.name]=f;cases.append({'name':name,'layer':layer,'image':image,'mode':mode,'input':f.name,'graph':f'/tmp/opi-vit-full-r1/graphs/layer{layer:02d}.nb'})
for image in [1,2]:cases.append({'name':f'w8-i{image}','layer':0,'image':image,'mode':'w8-reference','input':f'l00-i{image}-reference.f32','graph':remote+'/w8.nb'})
(P/'isolate-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
with b.hostlock():
 locked('mkdir '+remote+' && mkdir '+remote+'/results',check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/probe',check=True)
 for c in cases:
  cmd='cd '+remote+' && taskset -c 7 ./probe '+shlex.quote(c['graph'])+' '+c['input']+' results/'+c['name']+'.f32 results/'+c['name']+'.json 257 > results/'+c['name']+'.log 2>&1'
  z=locked(cmd,capture_output=True,timeout=90);assert z.returncode==0,(c,z.returncode)
  (P/'isolate-status.json').write_text(json.dumps({'status':'running','last':c['name'],'time':time.time()}))
 data=locked('cd '+remote+' && tar czf - results',capture_output=True,check=True,timeout=90).stdout
(P/'isolate-results.tar.gz').write_bytes(data)
with tarfile.open(P/'isolate-results.tar.gz') as t:t.extractall(O,filter='data')
(P/'isolate-status.json').write_text(json.dumps({'status':'complete','cases':len(cases)}))
print('complete',len(cases))
