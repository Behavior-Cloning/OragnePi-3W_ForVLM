from pathlib import Path
import json,hashlib,urllib.request
P=Path(__file__).resolve().parent
(P/'images').mkdir(exist_ok=True)
for c in json.loads((P/'vqa-cases.json').read_text())['cases']:
 f=P/'images'/(c['id']+'.jpg')
 if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==c['image_sha256']:continue
 assert c['image_url'].startswith('https://s3.amazonaws.com/images.cocodataset.org/val2014/')
 with urllib.request.urlopen(c['image_url'],timeout=45) as response:data=response.read()
 assert hashlib.sha256(data).hexdigest()==c['image_sha256'],c['id']
 tmp=f.with_suffix('.jpg.part');tmp.write_bytes(data);tmp.replace(f)
 print(c['id'])
