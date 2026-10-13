"""Read-only evidence capture when root TF I/O stalls; write only on host."""
import json
import os
from pathlib import Path
import subprocess
import shlex

code=r'''
import json,os,re,time
from pathlib import Path
fd=os.open('/dev/kmsg',os.O_RDONLY|os.O_NONBLOCK);lines=[]
try:
    while True:lines.append(os.read(fd,16384).decode(errors='replace'))
except BlockingIOError:pass
finally:os.close(fd)
text=''.join(lines)
relevant=[line for line in text.splitlines() if re.search(r'mmc|sdmmc|ext4|EXT4|dma hold|pwm|cooling_device|I/O error',line)]
blocked=[];experiment=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        status=(p/'status').read_text();name=(p/'comm').read_text().strip()
        if 'State:\tD' in status:
            blocked.append({'pid':int(p.name),'name':name,'status':status,'stack':(p/'stack').read_text(),'wchan':(p/'wchan').read_text()})
        if name in ('python3','llama-server','thermal_heater'):
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            experiment.append({'pid':int(p.name),'name':name,'cmdline':cmd,'wchan':(p/'wchan').read_text()})
    except OSError:pass
result={'timestamp':time.time(),'temperature_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()),
        'meminfo':Path('/proc/meminfo').read_text(),'diskstats':Path('/proc/diskstats').read_text(),
        'relevant_kernel_log':relevant[-120:],'blocked_processes':blocked,'experiment_processes':experiment,
        'scope':'read-only; no restart, unbind, fsck or thermal policy change'}
print(json.dumps(result))
'''
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=5','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','root@192.168.88.29']
r=subprocess.run(ssh+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30,check=True)
d=json.loads(r.stdout)
out=Path(__file__).resolve().parent/'evidence-thermal-20261013/storage-fault.json'
out.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'path':str(out),'temperature_mC':d['temperature_mC'],'blocked':[(r['pid'],r['name'],r['wchan']) for r in d['blocked_processes']]}))
