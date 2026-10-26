from pathlib import Path
import sys,json,hashlib,tarfile
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit-neoncopy-r1';out=P/'neoncopy-r1';out.mkdir(exist_ok=False)
cfg=json.loads((P/'tower-r2/config.json').read_text());cfg['sha256']['tower-probe']=hashlib.sha256((P/'tower-neon-probe').read_bytes()).hexdigest()
(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
with b.hostlock():
 locked('mkdir '+remote,check=True);upload(P/'tower-neon-probe',remote+'/tower-probe');upload(out/'config.json',remote+'/config.json');locked('chmod +x '+remote+'/tower-probe',check=True)
 z=locked(remote+'/tower-probe --selftest',capture_output=True,text=True,check=True,timeout=90);(out/'selftest.txt').write_text(z.stdout)
 for name,mode in [('base',1),('fast',2),('repeat',1)]:
  path=remote+'/'+name;locked('mkdir '+path+' && ln -s /tmp/opi-vit-tower-r2/graphs '+path+'/graphs && cp '+remote+'/tower-probe '+remote+'/config.json /tmp/opi-vit-tower-r2/board_tower.py /tmp/opi-vit-tower-r2/input1.f32 /tmp/opi-vit-tower-r2/input2.f32 '+path+'/',check=True)
  z=locked('cd '+path+' && VIT_FAST_COPY='+str(mode)+' python3 board_tower.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
  data=locked('cd '+path+' && tar czf - results controller.log',capture_output=True,check=True,timeout=90).stdout
  (out/(name+'.tar.gz')).write_bytes(data)
  with tarfile.open(out/(name+'.tar.gz')) as t:t.extractall(out/name,filter='data')
  print(name,z.returncode,flush=True);assert z.returncode==0
