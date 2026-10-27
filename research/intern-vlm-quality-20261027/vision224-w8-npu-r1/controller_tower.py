from pathlib import Path
import sys,json,hashlib,tarfile
P=Path(__file__).resolve().parent;R=P.parents[2]
sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
state=json.loads((P/'build-status.json').read_text());assert state['state']=='compiled-not-board-validated' and len(state['cases'])==24
S=P.parent/'vit-full-r1/endpoint-assets';remote='/tmp/opi-vision224-w8-npu-r1';out=P/'board';out.mkdir(exist_ok=False)
files={'tower-probe':P.parent/'vit-full-r1/tower-fast16-probe','board_tower.py':P/'board_tower.py','input1.f32':S/'stem1.f32','input2.f32':S/'stem2.f32'}
for c in state['cases']:
 f=Path(c['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==c['sha256'];files['graphs/'+c['name']+'.nb']=f
files['graphs/stem.nb']=P.parent/'vit-full-r1/endpoints-nbg/stem.nb'
files['graphs/projector.nb']=P.parent/'vit-full-r1/selective-projector-norm-gelu.nb'
hashes={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}
(out/'config.json').write_text(json.dumps({'sha256':hashes,'scope':'NPU224 normalized image -> stem +24 W8A16/FP16 ViT + official projector ->64x896; sequential NPU language tested separately'},indent=2));files['config.json']=out/'config.json'
with b.hostlock():
 pre=locked('cat /proc/meminfo; df -B1 /tmp; cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,check=True).stdout;(out/'pre-health.txt').write_text(pre)
 import re
 assert int(re.search(r'MemAvailable:\s+(\d+)',pre).group(1))*1024 > sum(f.stat().st_size for f in files.values())+1100*1024**2
 free=int(locked('df -B1 --output=avail /tmp | tail -1',capture_output=True,text=True,check=True).stdout);assert free>sum(f.stat().st_size for f in files.values())+96*1024**2
 locked('test ! -e '+remote+' && mkdir '+remote+' && mkdir '+remote+'/graphs',check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/tower-probe',check=True)
 z=locked('cd '+remote+' && python3 board_tower.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 raw=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,check=True,timeout=90).stdout;(out/'raw.tar.gz').write_bytes(raw)
 with tarfile.open(out/'raw.tar.gz') as t:
  assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(out)
 post=locked('cat /proc/meminfo; cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,check=True).stdout;(out/'post-health.txt').write_text(post)
 print('tower return',z.returncode);assert z.returncode==0
 # Archive already verified on Windows; remove only this run's large NBG copies before language load.
 expected=''.join(hashes[n]+'  '+remote+'/'+n+'\n' for n in hashes if n.startswith('graphs/'))
 locked('sha256sum -c -',input=expected,text=True,capture_output=True,check=True,timeout=90)
 locked('rm -- '+' '.join(remote+'/'+n for n in hashes if n.startswith('graphs/')),check=True)
 (out/'board-graphs-released.json').write_text(json.dumps({'archived_on_windows':True,'deleted_only_run_graphs':list(n for n in hashes if n.startswith('graphs/'))}))
