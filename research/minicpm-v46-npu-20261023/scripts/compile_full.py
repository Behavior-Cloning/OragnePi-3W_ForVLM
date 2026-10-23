from pathlib import Path
import subprocess,json,time,concurrent.futures,threading
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');lock=threading.Lock();results=[]
for _ in range(120):
 if (r/'results/export-full.json').exists():break
 time.sleep(2)
else:raise RuntimeError('export not complete')
meta=json.loads((r/'results/export-full.json').read_text());assert meta['reference_max_abs']<.02
names=['l0','l3']+['l'+str(i) for i in range(24) if i not in [0,3]]+['head'+str(i) for i in range(8)]
def build(name):
 src=b/'full'/name;out=b/'compiled'/name;out.parent.mkdir(exist_ok=True)
 cmd=['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(src),'--results-root',str(b),'--output',str(out),'--dtype','fp16','--timeout','240']
 start=time.time()
 with (r/'results'/('full-compile-'+name+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 row={'name':name,'exit':z.returncode,'seconds':time.time()-start}
 with lock:results.append(row);(r/'results/full-compile-status.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
 return row
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(build,names))
print('COMPLETE',flush=True)
