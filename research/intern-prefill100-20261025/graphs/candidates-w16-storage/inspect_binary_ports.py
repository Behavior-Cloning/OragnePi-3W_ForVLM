from pathlib import Path
import subprocess,fcntl,json
P=Path(__file__).resolve().parent;SDK='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator';env={'LD_LIBRARY_PATH':SDK+'/lib:'+SDK+'/../common/lib','VSIMULATOR_CONFIG':'VIP9000NANODI_PLUS_PID0X1000003B','VSIMULATOR_SHADER_CORE_COUNT':'1'}
base=['docker','run','--rm','--network','none','-v',str(P)+':/work','-w','/work']
for k,v in env.items():base+=['-e',k+'='+v]
with open('/tmp/opi-prefill-compile.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 cmd=base+['--entrypoint','gcc','ubuntu-npu:v2.0.10.2','-std=c99','-O2','inspect_nbg.c','-I'+SDK+'/include','-L'+SDK+'/lib','-L'+SDK+'/../common/lib','-lOpenVX','-lGAL','-lvdtproxy','-o','inspect_nbg']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(P/'inspect-compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 for v in ['official-export']:
  cmd=base+['--entrypoint','./inspect_nbg','ubuntu-npu:v2.0.10.2','/work/'+v+'/network_binary.nb'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
  (P/(v+'-binary-ports.json')).write_text(r.stdout);(P/(v+'-binary-ports.stderr')).write_text(r.stderr);print(v,r.returncode,r.stdout[:400],r.stderr[:400]);assert r.returncode==0,r.stderr;assert json.loads(r.stdout)['count']==16
