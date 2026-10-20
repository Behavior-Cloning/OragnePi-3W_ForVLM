import concurrent.futures,hashlib,json,time,urllib.request,urllib.parse,sys
from pathlib import Path
root=Path(sys.argv[1]);token=(Path.home()/".config/jev-task/modelscope-token").read_text().strip()
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""):h.update(x)
 return h.hexdigest()
def download(repo,row):
 fn=row["Path"];dest=root/repo.replace("/","--")/fn;dest.parent.mkdir(parents=True,exist_ok=True);part=Path(str(dest)+".part")
 sha=row.get("Sha256");size=row["Size"]
 if dest.exists() and dest.stat().st_size==size and (not sha or digest(dest)==sha):return dict(file=fn,bytes=size,sha256=digest(dest),revision=row["Revision"])
 url="https://modelscope.cn/api/v1/models/"+repo+"/repo?"+urllib.parse.urlencode({"Revision":row["Revision"],"FilePath":fn})
 for attempt in range(5):
  try:
   pos=part.stat().st_size if part.exists() else 0
   if pos>size:part.unlink();pos=0
   if pos<size:
    headers={"Authorization":"Bearer "+token}
    if pos:headers["Range"]="bytes=%d-"%pos
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
     if pos and response.status!=206:pos=0
     with part.open("ab" if pos else "wb") as f:
      for chunk in iter(lambda:response.read(8*1024*1024),b""):f.write(chunk)
   actual=digest(part)
   if part.stat().st_size!=size or (sha and actual!=sha):
    part.unlink();raise ValueError("size or SHA256 mismatch: "+fn)
   part.replace(dest);print(repo,fn,"verified",size,flush=True)
   return dict(file=fn,bytes=size,sha256=actual,revision=row["Revision"])
  except Exception as e:
   print(repo,fn,"attempt",attempt+1,type(e).__name__,flush=True)
   if attempt==4:raise
   time.sleep(3*(attempt+1))
for repo in (sys.argv[2:] or ["autotrust/JEV-9B","autotrust/JEV-27B-VL"]):
 meta=json.loads((root/(repo.replace("/","--")+"-ms-files.json")).read_text())
 rows=[r for r in meta["Data"]["Files"] if r["Type"]=="blob"]
 manifest={"repo":repo,"source":"ModelScope","purpose":"download archive only; no GPU required","status":"running","files":[],"errors":[]}
 m=root/repo.replace("/","--")/"download-manifest.json";m.parent.mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures={pool.submit(download,repo,r):r for r in rows}
  for f in concurrent.futures.as_completed(futures):
   try:manifest["files"].append(f.result())
   except Exception as e:manifest["errors"].append({"file":futures[f]["Path"],"error":str(e)})
   m.write_text(json.dumps(manifest,indent=2))
 manifest["status"]="complete" if not manifest["errors"] else "failed"
 manifest["bytes"]=sum(f["bytes"] for f in manifest["files"]);m.write_text(json.dumps(manifest,indent=2))
 print(repo,manifest["status"],manifest["bytes"],flush=True)
