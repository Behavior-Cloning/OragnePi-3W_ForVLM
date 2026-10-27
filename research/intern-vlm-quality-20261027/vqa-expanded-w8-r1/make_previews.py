from pathlib import Path
from PIL import Image,ImageDraw
import json
P=Path(__file__).resolve().parent;cases=json.loads((P/'vqa-cases.json').read_text())['cases'];out=P/'previews';out.mkdir(exist_ok=True)
for start in [0,30]:
 sheet=Image.new('RGB',(1000,1260),'white');draw=ImageDraw.Draw(sheet)
 for k,c in enumerate(cases[start:start+30]):
  im=Image.open(P/'images'/(c['id']+'.jpg')).convert('RGB');im.thumbnail((190,180));x=(k%5)*200;y=(k//5)*210;sheet.paste(im,(x,y+25));draw.text((x+3,y+4),f"{c['index']:02d} {c['category']} {c['answer_en']}",fill='black')
 sheet.save(out/f'contact-{start}.jpg')
for c in cases:
 im=Image.open(P/'images'/(c['id']+'.jpg')).convert('RGB')
 for side in [112,224]:im.resize((side,side),Image.Resampling.BICUBIC).save(out/(c['id']+f'-{side}.png'))
