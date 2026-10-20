import json,hashlib,urllib.request,urllib.parse,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;token=Path('/root/.config/jev-task/modelscope-token').read_text().strip()
for alias,repo in [('laya','convaiinnovations/laya-multilingual'),('onejev08','OmniJev/OneJev1.6-0.8B')]:
 meta=json.loads((root/'sources'/('ms-'+repo.split('/')[-1]+'.json')).read_text());dest=root/'models'/alias;dest.mkdir(parents=True,exist_ok=True);manifest={'repo':repo,'source':'ModelScope','files':[]}
 try:
  for row in meta['Data']['Files']:
   fn=row['Path']
   if row['Type']!='blob' or fn.startswith('assets/') or fn=='.gitattributes':continue
   p=dest/fn;p.parent.mkdir(parents=True,exist_ok=True);sha=row.get('Sha256');rev=row['Revision']
   if p.exists() and p.stat().st_size==row['Size'] and (not sha or hashlib.sha256(p.read_bytes()).hexdigest()==sha):pass
   else:
    url='https://modelscope.cn/api/v1/models/'+repo+'/repo?'+urllib.parse.urlencode({'Revision':rev,'FilePath':fn})
    for attempt in range(3):
     try:
      req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token})
      h=hashlib.sha256();total=0
      with urllib.request.urlopen(req,timeout=90) as f,Path(str(p)+'.part').open('wb') as g:
       while True:
        buf=f.read(4*1024*1024)
        if not buf:break
        g.write(buf);h.update(buf);total+=len(buf)
      assert total==row['Size'],(fn,total,row['Size'])
      assert not sha or h.hexdigest()==sha,(fn,'hash mismatch')
      Path(str(p)+'.part').replace(p);break
     except Exception:
      if attempt==2:raise
      time.sleep(3)
   manifest['files'].append({'file':fn,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'revision':rev});print(alias,fn,'OK',flush=True)
  manifest['status']='complete'
 except Exception as e:manifest['status']='failed';manifest['error']=repr(e);print(alias,repr(e),flush=True)
 (dest/'download-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
