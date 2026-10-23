from pathlib import Path
import subprocess,time,os
r=Path(__file__).resolve().parents[1]
for _ in range(900):
 if not Path('/proc/48679').exists():break
 time.sleep(2)
else:raise RuntimeError('previous controller still running')
assert (r/'results/w8cal-runtime.jsonl').exists(),'base run missing'
for label,args in [('validate-base',[str(Path('/opt/opi-lab/minicpm-v46-npu-20261023/ref311/bin/python')),str(r/'scripts/validate_w8_generation.py'),'epssafe','w8cal']),('dfp-probes',['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/run_w8cal_probes.py'),'w8dfp']),('scaled-probe',['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/run_w8cal_probes.py'),'w8scaled-all']),('scaled-deploy',['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/deploy_w8scaled.py')])]:
 print('START',label,flush=True)
 with (r/'results'/('continuation-'+label+'.log')).open('w') as f:z=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print('END',label,z.returncode,flush=True)
 if z.returncode:raise RuntimeError(label+' failed')
print('COMPLETE',flush=True)
