import pathlib,urllib.request,urllib.parse,json,hashlib,concurrent.futures,time,shutil
P=pathlib.Path(__file__).parent
TOKEN=pathlib.Path('/root/.config/jev-task/modelscope-token').read_text().strip()
items=[]
for mid,mf,subdir,choose in [('google/paligemma2-3b-mix-224','google_paligemma2-3b-mix-224-files.json','model',lambda f:f['Size']<20000000 or f['Path'].endswith('.safetensors')),('bartowski/FuseChat-Llama-3.2-1B-Instruct-GGUF','bartowski_FuseChat-Llama-3.2-1B-Instruct-GGUF-files.json','llama-zh',lambda f:f['Path'].endswith('Q4_K_M.gguf') or f['Path']=='README.md')]:
 for f in json.loads((P/mf).read_text())['Data']['Files']:
  if f['Type']=='blob' and choose(f):items.append((mid,f,P/subdir/f['Path']))
assert sum(f['Size'] for _,f,_ in items)<7.0*1024**3
print('download planned',sum(f['Size'] for _,f,_ in items),flush=True)
def one(item):
 mid,f,dst=item;dst.parent.mkdir(parents=True,exist_ok=True);part=dst.with_name(dst.name+'.part');expected=f['Sha256']
 def sha(q):
  h=hashlib.sha256()
  with open(q,'rb') as s:
   for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
  return h.hexdigest()
 if dst.exists() and dst.stat().st_size==f['Size'] and sha(dst)==expected:return dict(model=mid,file=f['Path'],ok=True,existing=True,sha256=expected,size=f['Size'])
 url='https://modelscope.cn/api/v1/models/'+mid+'/repo?Revision='+f['Revision']+'&FilePath='+urllib.parse.quote(f['Path'])
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0
   if shutil.disk_usage(P).free-(f['Size']-offset)<8*1024**3:raise RuntimeError('F free-space guard 8GiB')
   headers={'Authorization':'Bearer '+TOKEN}
   if offset:headers['Range']='bytes='+str(offset)+'-'
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as r:
    if offset and r.status!=206:raise RuntimeError('server refused resume; partial preserved')
    with open(part,'ab') as out:
     while True:
      b=r.read(8*1024*1024)
      if not b:break
      out.write(b)
   if part.stat().st_size!=f['Size']:raise RuntimeError('wrong size '+str(part.stat().st_size))
   got=sha(part)
   if got!=expected:raise RuntimeError('hash mismatch')
   part.rename(dst);print('verified',f['Path'],f['Size'],flush=True)
   return dict(model=mid,file=f['Path'],ok=True,sha256=expected,size=f['Size'],revision=f['Revision'])
  except Exception as e:
   print('retry',f['Path'],attempt,type(e).__name__,str(e),flush=True);time.sleep(2)
 return dict(model=mid,file=f['Path'],ok=False,size=f['Size'],partial_bytes=part.stat().st_size if part.exists() else 0)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 out=list(ex.map(one,items))
(P/'download-result.json').write_text(json.dumps(out,indent=2))
print('complete',sum(x['ok'] for x in out),len(out),flush=True)
