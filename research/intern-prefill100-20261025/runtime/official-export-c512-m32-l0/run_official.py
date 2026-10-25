#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,subprocess,time,fcntl,shutil
P=Path(__file__).resolve().parent;SDK='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((P/'source-proof.json').read_text());s={'status':'waiting_compile_lock','started':time.time(),'sources':proof,'script_sha256':sha(__file__),'variants':{}}
def save():(P/'status.json').write_text(json.dumps(s,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s['lock_acquired']=time.time();save()
 for v in ['plain','bitcast']:
  E=P/v;assert shutil.disk_usage('/mnt/f').free>8*1024**3
  assert sha(E/'vnn_decoder.c')==proof['variants'][v]['c_sha256'] and sha(E/'decoder.export.data')==proof['variants'][v]['data_sha256']
  assert sha(E/'main.c')==proof['main_sha256'] and sha(E/'makefile.linux')==proof['make_sha256']
  assert all(sha(E/x['file'])==x['sha256'] for x in proof['inputs'])
  env={'VIVANTE_SDK_DIR':SDK,'LD_LIBRARY_PATH':SDK+'/lib:'+SDK+'/../common/lib:'+SDK+'/lib/x64_linux:'+SDK+'/lib/x64_linux/vsim','VSIMULATOR_CONFIG':'VIP9000NANODI_PLUS_PID0X1000003B','VSIMULATOR_SHADER_CORE_COUNT':'1','VIV_VX_DUMP_NODE_TRACE_DB':'1','TF_CPP_MIN_LOG_LEVEL':'3','OMP_NUM_THREADS':'4','VIV_VX_ENABLE_SAVE_NETWORK_BINARY':'1','VIV_VX_SAVE_NETWORK_BINARY_PATH':'/work/network_binary.nb','VIV_VX_NBG_NAME':'decoder','VIV_VX_NBG_INPUT_RANK':'NCHW','VNN_LOOP_TIME':'1'}
  record={'environment':env,'commands':[]};s['variants'][v]=record;s['status']='running_'+v;save()
  for stage,args in [('make',['make','-f','makefile.linux','-j4']),('generate',['./gen_nbg','decoder.export.data']+[x['file'] for x in proof['inputs']])]:
   name='opi-official-'+v+'-'+str(int(time.time()));cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','4','--memory','8g','-v',str(E)+':/work','-w','/work']
   for k,val in env.items():cmd+=['-e',k+'='+val]
   cmd+=['--entrypoint',args[0],'ubuntu-npu:v2.0.10.2']+args[1:];record['commands'].append(cmd);record['stage']=stage;save();t=time.monotonic()
   with (P/(v+'-'+stage+'.log')).open('wb') as log:
    try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','-t','2',name],stdout=log,stderr=subprocess.STDOUT,timeout=15);rc=124
   record[stage]={'rc':rc,'seconds':time.monotonic()-t,'log_sha256':sha(P/(v+'-'+stage+'.log'))};save()
   if rc:break
  if rc:s['status']='failed_'+v+'_'+stage;break
  nb=E/'network_binary.nb'
  if not nb.exists():s['status']='failed_'+v+'_no_nbg';break
  record['nbg']={'path':str(nb),'bytes':nb.stat().st_size,'sha256':sha(nb),'first32bytes_hex':nb.read_bytes()[:32].hex()};record['state']='generated';save()
 else:s['status']='both_generated_pending_nbg_port_and_payload_audit'
 s['finished']=time.time();s['free_F_bytes']=shutil.disk_usage('/mnt/f').free;save()
print(json.dumps(s))
