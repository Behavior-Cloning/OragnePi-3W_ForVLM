from pathlib import Path
import subprocess,sys,shlex
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;d='/opt/opi-vlm/jev-lab-20261020'
with b.hostlock():
 for n in ['board_optimize.py','board_adapters.py','jev_style_decision_board.py']:b.scp(r/n,b.HOST+':'+d+'/'+n,check=True)
 for name in ['nano33-fp32.onnx','nano33-int8.onnx']:b.scp(r/'models/nano33'/name,b.HOST+':'+d+'/models/nano33/'+name,check=True,timeout=180)
 b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock '+d+'/venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime==1.23.1',timeout=300,check=True)
 laya_ort=(r/'results/laya-onnx-export.json').exists()
 if laya_ort:b.scp(r/'models/laya/laya-fp32.onnx',b.HOST+':'+d+'/models/laya/laya-fp32.onnx',check=True,timeout=240)
 laya_int8=(r/'results/laya-onnx-int8-parity.json').exists()
 if laya_int8:b.scp(r/'models/laya/laya-int8.onnx',b.HOST+':'+d+'/models/laya/laya-int8.onnx',check=True,timeout=240)
 for model in ['jev08','laya','nano33']:
  configs=[('a76-t2','6,7',2,{}),('a76-t1','6',1,{}),('a55-t2','0,1',2,{})]
  if model in ['nano33','laya']:configs.append(('a76-int8','6,7',2,{'JEV_DYNAMIC_INT8':'1'}))
  if model=='laya' and laya_ort:configs.append(('a76-ort-fp32','6,7',2,{'LAYA_ORT':'fp32'}))
  if model=='laya' and laya_int8:configs.append(('a76-ort-int8','6,7',2,{'LAYA_ORT':'int8'}))
  if model=='nano33':configs.extend([('a76-batch1','6,7',2,{'NANO_BATCH':'1'}),('a76-ort-fp32','6,7',2,{'NANO_ORT':'fp32'}),('a76-ort-int8','6,7',2,{'NANO_ORT':'int8'})])
  if model=='jev08':configs.extend([('a76-seq2','6,7',2,{'JEV_N_SEQ_MAX':'2'}),('a76-seq2-ub256','6,7',2,{'JEV_N_SEQ_MAX':'2','JEV_UBATCH':'256'})])
  # Probe faster runtimes early; all declared configs remain in the matrix.
  order={'a76-t2':0,'a76-seq2':1,'a76-seq2-ub256':2,'a76-ort-fp32':1,'a76-ort-int8':2,'a76-int8':3,'a76-t1':4,'a55-t2':5,'a76-batch1':6}
  configs.sort(key=lambda c:order[c[0]])
  for label,cores,threads,env in configs:
   cool="import time;from pathlib import Path\nfor i in range(60):\n t=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))\n if t<70:break\n if i%6==0:print('cooldown',t,flush=True)\n time.sleep(5)\nelse:raise RuntimeError('did not cool below 70C')"
   b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock '+d+'/venv/bin/python -u -c '+shlex.quote(cool),timeout=320,check=True)
   cmd='env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false '+' '.join(k+'='+v for k,v in env.items())+' timeout --kill-after=15s 900s taskset -c '+cores+' '+d+'/venv/bin/python -u '+d+'/board_optimize.py --model '+model+' --threads '+str(threads)+' --label '+label
   result=b.ssh('flock -w 10 /run/opi-npu-kv-benchmark.lock bash -o pipefail -c '+shlex.quote(cmd+' 2>&1 | tee '+d+'/results/opt-'+model+'-'+label+'.log'),timeout=950);print(model,label,'exit',result.returncode,flush=True)
   subprocess.run(['sshpass','-e','rsync','-rt','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+d+'/results/',str(r/'results/board')+'/'],env=b.env(),check=True,timeout=180)
