from pathlib import Path
import json,urllib.request,urllib.parse,hashlib,time
r=Path(__file__).resolve().parents[1];m=Path('/mnt/f/Files/OPi-Lab/models/MiniCPM-V-4.6')
files=json.loads((r/'sources/modelscope-files.json').read_text())['Data']['Files'];report=[]
for f in sorted(files,key=lambda v:v['Size']):
 if f['Type']!='blob':continue
 p=m/f['Path'];p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 u='https://modelscope.cn/api/v1/models/OpenBMB/MiniCPM-V-4.6/repo?'+urllib.parse.urlencode({'Revision':f['Revision'],'FilePath':f['Path']})
 row={'path':f['Path'],'revision':f['Revision'],'sha256_expected':f['Sha256'],'bytes_expected':f['Size']};report.append(row)
 try:
  start=time.monotonic();h=hashlib.sha256();n=0
  with urllib.request.urlopen(u,timeout=120) as res,tmp.open('wb') as out:
   while True:
    block=res.read(4*1024*1024)
    if not block:break
    out.write(block);h.update(block);n+=len(block)
    if n%(128*1024*1024)==0:print(f['Path'],n,flush=True)
  row.update(bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-start)
  assert n==f['Size'] and h.hexdigest()==f['Sha256'],'size/hash mismatch'
  tmp.rename(p);row['verified']=True
 except Exception as e:row.update(verified=False,error=repr(e));print(row,flush=True)
 (r/'results/download-manifest.json').write_text(json.dumps(report,indent=2))
 print(f['Path'],row.get('verified'),flush=True)
