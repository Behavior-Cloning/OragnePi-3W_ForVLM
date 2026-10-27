from pathlib import Path
import json,urllib.request,hashlib,concurrent.futures
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;groups=json.loads((P/'candidates.json').read_text());(P/'images').mkdir(exist_ok=True);(P/'previews').mkdir(exist_ok=True)
def fetch(c):
 f=P/'images'/(c['id']+'.jpg')
 try:
  if not f.exists():
   with urllib.request.urlopen(c['image_url'],timeout=45) as r:data=r.read()
   f.write_bytes(data)
  im=Image.open(f);im.verify();return {'id':c['id'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'ok':True}
 except Exception as e:return {'id':c['id'],'ok':False,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,[c for cs in groups.values() for c in cs]))
(P/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n');print('downloaded',sum(r['ok'] for r in rows),'of',len(rows),flush=True)
for group,cs in groups.items():
 canvas=Image.new('RGB',(1200,220*((len(cs)+5)//6)), 'white');draw=ImageDraw.Draw(canvas)
 for i,c in enumerate(cs):
  f=P/'images'/(c['id']+'.jpg')
  if not f.exists():continue
  im=Image.open(f).convert('RGB');im.thumbnail((196,190));x=(i%6)*200;y=(i//6)*220;canvas.paste(im,(x,y+24));draw.text((x,y),str(i)+' '+c['answer_en'][:24],fill='black')
 canvas.save(P/'previews'/(group+'.jpg'),quality=90)
