from pathlib import Path
import json,hashlib,shutil,subprocess,time,fcntl,os
import yaml
P=Path(__file__).resolve().parent;S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vitfull-20261027-r1');state={'status':'compiling','pid':os.getpid(),'cases':[]}
def save():(P/'compile-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for layer in [0,12]:
  for mode in ['input8','input-output8']:
   base=S/f'w8-native224-r1-l{layer:02d}';out=S/f'a8-screen-r1-l{layer:02d}-{mode}';out.mkdir(exist_ok=False)
   for n in ['decoder.json','decoder.data','tensor-inputmeta.yml']:shutil.copy2(base/n,out/n)
   graph=json.loads((base/'decoder.json').read_text())['Layers'];q=yaml.safe_load((base/'decoder.quantize').read_text());qp=q['quantize_parameters'];changed=[]
   for name,node in graph.items():
    if node['op']!='fullconnect':continue
    keys=[node['inputs'][0]]+(['@'+name+':out0'] if mode=='input-output8' else [])
    for k in keys:
     old=qp[k].copy();assert old['qtype']=='i16' and old['quantizer']=='dynamic_fixed_point'
     qp[k]={**old,'qtype':'i8','fl':old['fl']-8};changed.append({'node':k,'before':old,'after':qp[k]})
    # Let compiler infer bias metadata for the changed activation precision.
    qp.pop('@'+name+':bias',None)
   (out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
   name=f'opi-vit-a8-{layer}-{mode}';cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(S)+':/work','-w','/work/'+out.name,'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
   state.update(layer=layer,mode=mode);save();start=time.monotonic()
   with (P/f'l{layer:02d}-{mode}.log').open('w') as log:
    try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240);rc=z.returncode
    except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',name],stdout=log,stderr=subprocess.STDOUT);rc=124
   nb=out/'export_nbg_unify/network_binary.nb';case={'layer':layer,'mode':mode,'changes':changed,'returncode':rc,'seconds':time.monotonic()-start,'path':str(nb),'sha256':hashlib.sha256(nb.read_bytes()).hexdigest() if nb.exists() else None,'source_weight_sha256':hashlib.sha256((base/'decoder.data').read_bytes()).hexdigest()};state['cases'].append(case);save();print(layer,mode,rc,flush=True)
 state['status']='compiled-not-board-tested';save()
