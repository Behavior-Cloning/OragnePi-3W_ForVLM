import pathlib,urllib.request,urllib.parse,json,hashlib,time,shutil,sys,datetime
P=pathlib.Path(__file__).parent
TOKEN=pathlib.Path('/root/.config/jev-task/modelscope-token').read_text().strip()
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
f=json.loads((P/'source-file-record.json').read_text())
mid='bartowski/FuseChat-Llama-3.2-1B-Instruct-GGUF'
print('Planned Q8-only download bytes:',f['Size'],flush=True)
result=one((mid,f,P/f['Path']))
result.update(actual_download_source='ModelScope mirror',actual_request_url='https://modelscope.cn/api/v1/models/'+mid+'/repo?Revision='+f['Revision']+'&FilePath='+urllib.parse.quote(f['Path']),revision=f['Revision'],timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),upstream_model='FuseAI/FuseChat-Llama-3.2-1B-Instruct',quantized_repository=mid,same_source_revision_as_existing_Q4=True)
(P/'download-manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
sys.exit(0 if result['ok'] else 1)
