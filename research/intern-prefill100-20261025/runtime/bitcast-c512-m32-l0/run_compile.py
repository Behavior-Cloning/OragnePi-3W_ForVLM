#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,hashlib,time,fcntl,shutil
P=Path(__file__).resolve().parent;E=P/'export';SDK='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
status={'status':'waiting_compile_lock','started':time.time(),'source_sha256':sha(__file__),'main_sha256':sha(E/'main.c'),'decoder_sha256':sha(E/'vnn_decoder.c'),'scope':'WSL SDK compile/verify/export only. No board or hardware inference.'}
def save(): (P/'status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);status['lock_acquired']=time.time();status['status']='building';save()
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 env={'VIVANTE_SDK_DIR':SDK,'LD_LIBRARY_PATH':SDK+'/lib:'+SDK+'/../common/lib:'+SDK+'/lib/x64_linux:'+SDK+'/lib/x64_linux/vsim','VSIMULATOR_CONFIG':'VIP9000NANODI_PLUS_PID0X1000003B','VSIMULATOR_SHADER_CORE_COUNT':'1','VIV_VX_DUMP_NODE_TRACE_DB':'1','TF_CPP_MIN_LOG_LEVEL':'3','OMP_NUM_THREADS':'4'}
 base=['docker','run','--rm','--network','none','--cpus','4','--memory','8g','-v',str(E)+':/work','-w','/work']
 for k,v in env.items():base+=['-e',k+'='+v]
 stages=[('make',['make','-f','makefile.linux','-j4']),('verify-export',['./gen_nbg','decoder.export.data','network_binary.nb'])]
 status['environment']=env;status['commands']=[];save()
 for name,args in stages:
  cmd=base+['--entrypoint',args[0],'ubuntu-npu:v2.0.10.2']+args[1:];status['commands'].append(cmd);status['stage']=name;save();t=time.monotonic()
  with (P/(name+'.log')).open('wb') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
   except subprocess.TimeoutExpired:rc=124
  status[name]={'rc':rc,'seconds':time.monotonic()-t,'log_sha256':sha(P/(name+'.log'))};save()
  if rc:status['status']='failed_'+name;break
 else:
  nb=E/'network_binary.nb';status['status']='native_nbg_exported_unvalidated_on_board';status['nbg']={'path':str(nb),'bytes':nb.stat().st_size,'sha256':sha(nb),'first32bytes_hex':nb.read_bytes()[:32].hex()}
 status['finished']=time.time();status['free_F_bytes']=shutil.disk_usage('/mnt/f').free;save()
print(json.dumps(status))
