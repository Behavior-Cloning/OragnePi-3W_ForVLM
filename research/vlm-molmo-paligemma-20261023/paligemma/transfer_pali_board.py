import pathlib,sys,subprocess,shlex,json,time
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/vlm-new-models-20261023';src=P/'paligemma2-3b-mix-224-sfp.sbs'
with b.hostlock():
 code="import shutil;assert shutil.disk_usage('/opt').free>"+str(src.stat().st_size+1024**3)+";print(shutil.disk_usage('/opt').free)"
 b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code),timeout=20).check_returncode()
 r=b.ssh('command -v rsync',timeout=20,capture_output=True,text=True);r.check_returncode()
 rsh=shlex.join(['sshpass','-e','ssh',*b.COMMON])
 cmd=['rsync','--partial','--append-verify','--info=progress2','--rsync-path=flock -w 10 /run/opi-npu-kv-benchmark.lock rsync','-e',rsh,str(src),b.HOST+':'+B+'/'+src.name]
 print('transfer start',time.time(),flush=True);r=subprocess.run(cmd,env=b.env(),timeout=3600);r.check_returncode();print('transfer completed',flush=True)
 code="import hashlib,pathlib,json,shutil;f=pathlib.Path('"+B+"/"+src.name+"');h=hashlib.sha256();s=f.open('rb');[h.update(c) for c in iter(lambda:s.read(8388608),b'')];print(json.dumps(dict(bytes=f.stat().st_size,sha256=h.hexdigest(),free=shutil.disk_usage('/opt').free)))"
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code),timeout=100,capture_output=True,text=True);r.check_returncode();print(r.stdout,flush=True);(P/'sfp-board-manifest.json').write_text(r.stdout)
 d=json.loads(r.stdout);assert d['sha256']==json.loads((P/'sfp-manifest.json').read_text())['sha256'];assert d['free']>=1024**3
