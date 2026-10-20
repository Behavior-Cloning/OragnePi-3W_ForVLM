from pathlib import Path
import urllib.request,json,subprocess,hashlib
r=Path(__file__).resolve().parent;repo="sdmlai/nano-jev";rev="ffee345e97797bef0948561461ea54fe558588a0";dest=r/"models/nano33";dest.mkdir(parents=True,exist_ok=True)
meta=json.loads(subprocess.check_output(["curl","-fLsS","--max-time","25","https://hf-mirror.com/api/models/"+repo+"/revision/"+rev]));(r/"sources/nano33-metadata.json").write_text(json.dumps(meta,indent=2))
m={"repo":repo,"revision":rev,"source":"HF mirror; ModelScope same ID record not found","files":[],"status":"running"}
for row in meta["siblings"]:
 name=row["rfilename"]
 if name.startswith(".") or name.endswith(".png"):continue
 f=dest/name;f.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(["curl","-fL","--retry","3","--max-time","900","-C","-","-sS","https://hf-mirror.com/"+repo+"/resolve/"+rev+"/"+name,"-o",str(f)+".part"],check=True)
 part=Path(str(f)+".part");sha=hashlib.sha256(part.read_bytes()).hexdigest()
 if name=="model.safetensors":assert sha=="c8bccb52c1591a8de0c6e31f23bad3a558bba7daae4925c6810d477485f290cc"
 part.replace(f);m["files"].append(dict(file=name,bytes=f.stat().st_size,sha256=sha));print(name,"OK",flush=True)
m["status"]="complete";(dest/"download-manifest.json").write_text(json.dumps(m,indent=2))
