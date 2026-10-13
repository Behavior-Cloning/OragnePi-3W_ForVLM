"""Upload and retrieve an isolated, cached CPU reference on the development board."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--prompt-folder',type=Path,required=True)
p.add_argument('--model-id',choices=['internvl3-1b','smolvlm-500m'],required=True)
p.add_argument('--context',type=int,required=True)
p.add_argument('--tag',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--temperature-band-mc',type=int,nargs=2)
a=p.parse_args()
from board_host_lock import lock_board_from_host
lock_board_from_host()
assert a.tag.replace('-','').isalnum() and 1<=a.repeats<=20
prompt=json.loads((a.prompt_folder/'prompt.json').read_text())
assert len(prompt['token_ids'])+127<=a.context
assert not a.output.exists(), 'keep previous evidence immutable'
names={'internvl3-1b':'InternVL3-1B-Instruct-Q8_0.gguf','smolvlm-500m':'SmolVLM-500M-Instruct-Q8_0.gguf'}
prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
ssh=prefix+['ssh','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
remote='/opt/opi-vlm/npu-kv-probe/'+a.tag
subprocess.run(ssh+['mkdir -p '+remote],check=True,timeout=20)
subprocess.run(scp+[str(Path(__file__).with_name('cpu_incremental_reference.py')),
    str(Path(__file__).with_name('thermal_condition.py')),
    str(Path(__file__).with_name('thermal_heater.c')),
    str(a.prompt_folder/'prompt.json'),'root@192.168.88.29:'+remote+'/'],check=True,timeout=60)
command=['python3',remote+'/cpu_incremental_reference.py','--model','/opt/opi-vlm/models/'+a.model_id+'/'+names[a.model_id],
    '--prompt',remote+'/prompt.json','--cached-prefix',str(prompt['cached_prefix']),
    '--context',str(a.context),'--steps','128','--repeats',str(a.repeats),'--output',remote+'/result']
command.append('--capture-nonstream')
if a.temperature_band_mc:command+=['--temperature-band-mc',*map(str,a.temperature_band_mc)]
build='cc -O3 -Wall -Wextra -Werror -march=armv8-a+simd '+remote+'/thermal_heater.c -pthread -o '+remote+'/thermal_heater'
result_run=subprocess.run(ssh+['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(build+' && '+shlex.join(command))],timeout=3500)
a.output.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(scp+['-r','root@192.168.88.29:'+remote+'/result',str(a.output)],check=True,timeout=120)
result_run.check_returncode()
result=json.loads((a.output/'results.json').read_text())
assert result['status']=='complete' and result['prompt']==prompt
print(json.dumps(result['medians'],indent=2),flush=True)
