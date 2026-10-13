"""Bounded recovery evidence; does not clear the benchmark storage interlock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from board_host_lock import lock_board_from_host


REMOTE = r'''
import hashlib,json,os,re,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
expected=json.load(sys.stdin)
def blocked():
    result=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if 'State:\tD' in (p/'status').read_text():
                result.append({'pid':p.name,'name':(p/'comm').read_text().strip(),'wchan':(p/'wchan').read_text().strip()})
        except OSError:pass
    return result
def kernel():
    return subprocess.run(['dmesg'],capture_output=True,text=True,check=True,timeout=5).stdout
def relevant(log):
    return [line for line in log.splitlines() if re.search(r'4020000.sdmmc|mmcblk1|EXT4-fs|I/O error|Buffer I/O|dma hold',line)]
def runtime_errors(log):
    result=[]
    for line in relevant(log):
        stamp=re.match(r'\[\s*([0-9.]+)\]',line)
        if stamp and float(stamp[1])>10 and re.search(r'error|timeout|abort|read-only|blocked for',line,re.I):result.append(line)
    return result
data={'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
      'uptime_seconds':float(Path('/proc/uptime').read_text().split()[0]),
      'temperature_before_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()),
      'mounts':Path('/proc/mounts').read_text().splitlines(),
      'blocked_before':blocked()}
before=kernel();data['kernel_before']=relevant(before)
if data['blocked_before'] or runtime_errors(before):
    data['passed']=False;data['reason']='runtime storage errors or D-state before I/O check'
    print(json.dumps(data));sys.exit(0)
payload=os.urandom(1024*1024);start=time.monotonic()
with tempfile.NamedTemporaryFile(prefix='recovery-probe-',dir='/opt/opi-vlm/npu-kv-probe') as f:
    f.write(payload);f.flush();os.fsync(f.fileno())
    os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    f.seek(0);actual=f.read()
    data['write_read']={'bytes':len(payload),'sha256':hashlib.sha256(actual).hexdigest(),
                        'matches':actual==payload,'seconds':time.monotonic()-start,
                        'fsync':True,'cache_drop':'POSIX_FADV_DONTNEED on temporary file only'}
data['hashes']=[]
for item in expected:
    start=time.monotonic();h=hashlib.sha256()
    with open(item['path'],'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    data['hashes'].append(dict(item,actual=h.hexdigest(),matches=h.hexdigest()==item['sha256'],seconds=time.monotonic()-start))
after=kernel();data['kernel_after']=relevant(after)
data['runtime_storage_errors']=runtime_errors(after);data['blocked_after']=blocked()
data['fsck_journal']=subprocess.run(['journalctl','-b','-u','systemd-fsck-root.service','--no-pager'],capture_output=True,text=True,timeout=5).stdout
with urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=5) as response:
    health=json.load(response)
data['web_health']={key:health.get(key) for key in ('ready','busy','backend','model_id','metrics','error')}
data['temperature_after_mC']=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
data['passed']=data['write_read']['matches'] and all(i['matches'] for i in data['hashes']) and not data['runtime_storage_errors'] and not data['blocked_after'] and health['ready']
data['scope']='Small fsynced write/read and deployed-file hashes only; boot tuning timeouts preserved; no fsck repair or proof of long-term storage stability.'
print(json.dumps(data))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'preserve prior recovery evidence'
    lock_board_from_host()
    project=Path(__file__).resolve().parents[2]
    manifest=project/'npu/results/full-smol500-stable-v3-c128/w8-fused2-c768-splitkv-transk/board-aligned-smol500-h512-n32/deployed-sha256.txt'
    expected=[]
    for line in manifest.read_text().splitlines():
        sha,name=line.split(maxsplit=1)
        if name.endswith('.nb') or name=='embeddings.f32':
            expected.append({'path':'/opt/opi-vlm/npu-kv-probe/full-smol500-c768-aligned32/'+name,'sha256':sha})
    for model,name in [('smolvlm-500m','SmolVLM-500M-Instruct-Q8_0.gguf'),('internvl3-1b','InternVL3-1B-Instruct-Q8_0.gguf')]:
        with (project/'models'/model/name).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        expected.append({'path':'/opt/opi-vlm/models/'+model+'/'+name,'sha256':sha})
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','root@192.168.88.29']
    result=subprocess.run(ssh+['python3 -c '+shlex.quote(REMOTE)],input=json.dumps(expected),capture_output=True,text=True,timeout=180,check=True)
    data=json.loads(result.stdout);data['host_recorded_at']=time.time()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'passed':data['passed'],'boot_id':data['boot_id'],'files':len(data.get('hashes',[])),
                      'write_read':data.get('write_read'),'output':str(a.output)}),flush=True)
    if not data['passed']:raise RuntimeError('storage recovery gate failed; inspect evidence')


if __name__=='__main__':main()
