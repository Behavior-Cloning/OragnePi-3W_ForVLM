"""Record the inspected root-TF stall; no board action or inferred completion."""
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parents[1]/'results/thermal-aligned-20261013'
path=root/'state.json';state=json.loads(path.read_text())
state['status']='blocked-storage'
state['storage_pause']={'timestamp':time.time(),'reason':'SDMMC DMA hold timeout and persistent ext4_setattr/mmc_blk_rw_wait D-state at ~35 C',
    'evidence':'npu/kv-cache/evidence-thermal-20261013/storage-fault.json',
    'production_web':'online, ready, idle at read-only check','restarted':False,
    'experiment_SIGTERM_pids':[755531,755533],
    'host_queue_stopped':True,'scope':'filesystem I/O failure is not model latency or an accepted high-temperature run'}
entry=state['cases']['smol500-h512-n32']
entry['status']='blocked-storage';entry['bands']['high']['status']='blocked-storage'
for case,entry in state['cases'].items():
    for band,row in entry['bands'].items():
        if row.get('status')!='running':continue
        files=[f for f in root.glob(case+'-'+band+'-npu*.json') if json.loads(f.read_text()).get('status')=='complete']
        if files:
            f=max(files,key=lambda p:p.stat().st_mtime);d=json.loads(f.read_text())
            row.update(status='complete',performance=str(f),npu=d['medians']['batch32'])
    if entry['status']=='running':entry['status']='paused'
path.write_text(json.dumps(state,indent=2)+'\n')
