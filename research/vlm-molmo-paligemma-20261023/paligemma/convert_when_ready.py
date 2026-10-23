import pathlib,os,time,subprocess,shutil,json,hashlib
P=pathlib.Path(__file__).parent;G=P/'gemma.cpp-main'
while pathlib.Path('/proc/4436').exists() and not ((G/'python/configs.so').exists() and (G/'compression/python/compression.so').exists()):time.sleep(5)
if not (G/'compression/python/compression.so').exists():raise RuntimeError('Python extension build failed; see finish-runtime.log')
if shutil.disk_usage(P).free<13*1024**3:raise RuntimeError('F free-space guard before SFP conversion')
env=dict(os.environ,PYTHONPATH=str(P/'python-deps')+':'+str(G),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
cmd=['/opt/opi-lab/export-venv/bin/python',str(G/'python/convert_from_safetensors.py'),'--model_specifier=paligemma2-3b-224-sfp','--load_path='+str(P/'model/model.safetensors.index.json'),'--tokenizer_file='+str(P/'model/tokenizer.model'),'--metadata_file='+str(P/'conversion-metadata.csv'),'--sbs_file='+str(P/'paligemma2-3b-mix-224-sfp.sbs')]
print('conversion starting',flush=True);r=subprocess.run(cmd,env=env,cwd=G);print('conversion exit',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
f=P/'paligemma2-3b-mix-224-sfp.sbs';h=hashlib.sha256()
with f.open('rb') as q:
 for b in iter(lambda:q.read(8*1024*1024),b''):h.update(b)
(P/'sfp-manifest.json').write_text(json.dumps(dict(path=f.name,bytes=f.stat().st_size,sha256=h.hexdigest(),command=cmd,precision='SFP text linears; BF16 embedding and vision; F32 biases',conversion_exit=r.returncode),indent=2));print('SFP verified local',f.stat().st_size,h.hexdigest(),flush=True)
