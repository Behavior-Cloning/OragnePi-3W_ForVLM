from pathlib import Path
import sys,json,urllib.request,urllib.parse,hashlib,time
ROOT=Path(__file__).resolve().parent
r=ROOT/'models'/sys.argv[1];manifest=json.loads((r/'download-manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for f in manifest['files']:
 p=r/f['Path']
 if p.exists() and p.stat().st_size==f['Size'] and sha(p)==f['Sha256']:continue
 p.parent.mkdir(parents=True,exist_ok=True)
 url='https://modelscope.cn/models/'+manifest['repo']+'/resolve/'+f['Revision']+'/'+urllib.parse.quote(f['Path'])
 for attempt in range(3):
  try:
   temp=p.with_name(p.name+'.direct-part');total=0;last=time.monotonic()
   with urllib.request.urlopen(url,timeout=90) as src,temp.open('wb') as dst:
    while True:
     chunk=src.read(4*1024*1024)
     if not chunk:break
     dst.write(chunk);total+=len(chunk)
     if time.monotonic()-last>30:print('BOARD_DOWNLOAD',sys.argv[1],total,f['Size'],flush=True);last=time.monotonic()
   assert temp.stat().st_size==f['Size']
   assert sha(temp)==f['Sha256']
   temp.replace(p);print('BOARD_HASH_VERIFIED',f['Path'],flush=True);break
  except Exception as e:
   print('BOARD_DOWNLOAD_RETRY',repr(e),flush=True)
   if attempt==2:raise
print('BOARD_MODEL_READY',sys.argv[1],flush=True)
