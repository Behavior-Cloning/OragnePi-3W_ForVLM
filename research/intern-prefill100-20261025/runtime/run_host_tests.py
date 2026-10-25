from pathlib import Path
import subprocess,os,json,time
P=Path(__file__).resolve().parent
sdk='/opt/opi-lab/restored-slim3-20261017/board-runtime/usr/include'
results=[]
for test in ['test_schedule','test_release_ownership','test_two_prefill_banks','test_greedy']:
 exe=P/(test.replace('_','-')+'-host');cmd=['gcc','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+sdk,str(P/(test+'.c')),'-lm','-o',str(exe)]
 x=subprocess.run(cmd,text=True,capture_output=True);results.append({'kind':'build','test':test,'cmd':cmd,'rc':x.returncode,'stdout':x.stdout,'stderr':x.stderr})
 if x.returncode:break
 envs=[{}] if test!='test_two_prefill_banks' else [{},{'OPI_MASK_TEMPLATE':'1'},{'OPI_MASK_TEMPLATE':'1','OPI_MASK_TEMPLATE_VALIDATE':'1'},{'OPI_TEST_MASK_FP16':'1'},{'OPI_TEST_MASK_FP16':'1','OPI_MASK_TEMPLATE':'1'},{'OPI_TEST_MASK_FP16':'1','OPI_MASK_TEMPLATE':'1','OPI_MASK_TEMPLATE_VALIDATE':'1'}]
 for env in envs:
  t=time.monotonic();x=subprocess.run([str(exe)],text=True,capture_output=True,env={**os.environ,**env},timeout=90)
  results.append({'kind':'run','test':test,'env':env,'seconds':time.monotonic()-t,'rc':x.returncode,'stdout':x.stdout,'stderr':x.stderr[-4000:]})
  (P/'host-tests.json').write_text(json.dumps(results,indent=2));print(test,env,x.returncode,x.stdout.strip(),flush=True)
  if x.returncode:break
 if results[-1]['rc']:break
(P/'host-tests.json').write_text(json.dumps(results,indent=2))
