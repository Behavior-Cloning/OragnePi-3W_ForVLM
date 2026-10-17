import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
root=Path('/mnt/f/Files/OPi-Lab/audio-models-20261018/wheels');root.mkdir(exist_ok=True)
def get(item):
 name,version=item
 d=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30))
 candidates=[x for x in d['urls'] if 'aarch64' in x['filename'] and 'manylinux' in x['filename'] and (('cp310-cp310' in x['filename']) or ('py3-none' in x['filename']))]
 assert len(candidates)==1,(name,[x['filename'] for x in candidates])
 x=candidates[0];p=root/x['filename']
 if not p.exists():urllib.request.urlretrieve(x['url'],p)
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==x['digests']['sha256']
 r={'name':name,'version':version,'file':p.name,'url':x['url'],'sha256':h,'bytes':p.stat().st_size}
 print(json.dumps(r),flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(get,[('sherpa-onnx','1.13.8'),('sherpa-onnx-core','1.13.8'),('numpy','2.2.6')]))
(root.parent/'wheels.json').write_text(json.dumps(results,indent=2)+'\n')
