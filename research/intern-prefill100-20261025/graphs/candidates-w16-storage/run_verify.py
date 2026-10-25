#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,hashlib,time,fcntl,shutil
P=Path(__file__).resolve().parent;E=P/'export';SDK='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((P/'static-audit.json').read_text());assert sha(E/'vnn_decoder.c')==audit['candidate_c_sha256'];assert sha(E/'decoder.export.data')==audit['candidate_payload_sha256']
assert 'vsi_nn_GenerateNBG(' not in (E/'main.c').read_text()
status={'state':'waiting_compile_lock','started':time.time(),'source_sha256':sha(__file__),'main_sha256':sha(E/'main.c'),'decoder_sha256':sha(E/'vnn_decoder.c'),'scope':'One W16 storage candidate WSL SDK make/Verify only; no GenerateNBG, no board.'}
def save(): (P/'status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);status['lock_acquired']=time.time();status['state']='building';save()
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 env={'VIVANTE_SDK_DIR':SDK,'LD_LIBRARY_PATH':SDK+'/lib:'+SDK+'/../common/lib:'+SDK+'/lib/x64_linux:'+SDK+'/lib/x64_linux/vsim','VSIMULATOR_CONFIG':'VIP9000NANODI_PLUS_PID0X1000003B','VSIMULATOR_SHADER_CORE_COUNT':'1','VIV_VX_DUMP_NODE_TRACE_DB':'1','TF_CPP_MIN_LOG_LEVEL':'3','OMP_NUM_THREADS':'4'}
 base=['docker','run','--rm','--network','none','--cpus','4','--memory','8g','-v',str(E)+':/work','-w','/work']
 for k,v in env.items():base+=['-e',k+'='+v]
 status['environment']=env;status['commands']=[];save()
 for name,args in [('make',['make','-f','makefile.linux','-j4']),('verify',['./gen_nbg','decoder.export.data'])]:
  cmd=base+['--entrypoint',args[0],'ubuntu-npu:v2.0.10.2']+args[1:];status['commands'].append(cmd);status['stage']=name;save();t=time.monotonic()
  with (P/(name+'.log')).open('wb') as log:
   try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
   except subprocess.TimeoutExpired:rc=124
  status[name]={'rc':rc,'seconds':time.monotonic()-t,'log_sha256':sha(P/(name+'.log'))};save()
  if rc:status['state']='failed_'+name;break
 else:
  before=json.loads((E/'ports-before-verify.json').read_text());after=json.loads((E/'ports-after-verify.json').read_text());assert before==after
  assert len(after['inputs'])==11 and len(after['outputs'])==5
  status['actual_ports_before_after_exact']=True;status['state']='verify_passed_nbg_not_attempted'
 assert sha(E/'vnn_decoder.c')==audit['candidate_c_sha256'];assert sha(E/'decoder.export.data')==audit['candidate_payload_sha256']
 status['candidate_source_unchanged']=True;status['nb_generated']=False;status['finished']=time.time();status['free_F_bytes']=shutil.disk_usage('/mnt/f').free;save()
print(json.dumps(status))
