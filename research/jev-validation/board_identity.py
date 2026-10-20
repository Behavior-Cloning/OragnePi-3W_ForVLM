"""Read-only identity capture outside timed inference. Run under experiment lock."""
import hashlib,json,platform,importlib.metadata,os,subprocess
from pathlib import Path

def identity(root,models):
 root=Path(root); paths=set()
 for name in models:
  paths.update(p for p in (root/"models"/name).rglob("*") if p.is_file() and p.suffix not in (".pyc", ".log") and "__pycache__" not in p.parts)
 paths.update(p for p in (root/"bin").glob("*") if p.is_file())
 paths.update(p for p in root.glob("*.py") if p.is_file())
 paths.update(root/n for n in ("actions.json","cases.json","resident-inputs.json") if (root/n).exists())
 loader_resolution=None
 if "jev08" in models:
  z=subprocess.run(["ldd",str(root/"bin/jev-score")],capture_output=True,text=True,env={**os.environ,"LD_LIBRARY_PATH":str(root/"bin")+":"+os.environ.get("LD_LIBRARY_PATH","")},check=True)
  loader_resolution=z.stdout
  for line in z.stdout.splitlines():
   for word in line.split():
    if word.startswith("/") and Path(word).is_file():paths.add(Path(word))
 records=[]
 for p in sorted(paths):
  before=p.stat();h=hashlib.sha256()
  with p.open("rb") as stream:
   for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
  after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("Asset changed during hash: "+str(p))
  records.append(dict(path=str(p),bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=h.hexdigest()))
 return dict(platform=platform.platform(),models=models,loader_resolution=loader_resolution,files=records,packages={n:importlib.metadata.version(n) for n in ["torch","transformers","laya","onnxruntime"]},note="Computed after optimization and before final pairs. Does not retroactively identify earlier runs; NPU graph provenance remains in original experiment manifests.")
if __name__=="__main__":
 import sys
 print(json.dumps(identity(Path(__file__).resolve().parent,sys.argv[1:]),ensure_ascii=False,indent=2))
