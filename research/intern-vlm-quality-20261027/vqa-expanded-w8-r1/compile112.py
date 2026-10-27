from pathlib import Path
import subprocess,json,fcntl,hashlib,os
P=Path(__file__).resolve().parent;R=P.parents[2];S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vit112-native16-r1')
state={'state':'running','pid':os.getpid(),'scope':'24 ViT blocks 144 FC weights W8 per-channel; same INT16 activation boundaries and FP16 norm/GELU as validated tower; unchanged stem/projector','cases':[]}
def save():(P/'build-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 with open('/tmp/opi-prefill-compile.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for i in range(24):
   base=S/f'layer{i:02d}-compiled';out=S/f'w8-native112-r1-l{i:02d}';assert not out.exists()
   state['current']=i;save()
   cmd=['/opt/opi-lab/export-venv/bin/python',str(R/'npu/kv-cache/compile_weight_int8.py'),'--base',str(base),'--results-root',str(S),'--output',str(out),'--expected-linears','6','--weight-mode','perchannel','--timeout','240']
   with (P/f'layer{i:02d}.log').open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
   assert z.returncode==0,(i,z.returncode)
   nb=out/'export_nbg_unify/network_binary.nb';assert nb.exists()
   state['cases'].append({'name':f'layer{i:02d}','path':str(nb),'sha256':hashlib.sha256(nb.read_bytes()).hexdigest(),'bytes':nb.stat().st_size});save();print('W8 compiled',i,flush=True)
  state['state']='compiled-not-board-validated';save()
except BaseException as e:state.update(state='failed',error=str(e));save();raise
