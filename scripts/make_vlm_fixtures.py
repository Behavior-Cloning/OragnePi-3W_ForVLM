#!/usr/bin/env python3
"""Generate deterministic visual fixtures with known answers (Pillow on development host)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

target=Path(__file__).resolve().parents[1]/'docs/benchmarks/2026-10-09-system'
target.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
image=Image.new('RGB',(448,448),'white')
draw=ImageDraw.Draw(image)
draw.ellipse((35,55,145,165),fill='#e22626')
draw.ellipse((165,55,275,165),fill='#e22626')
draw.rectangle((80,245,180,345),fill='#1c57dc')
draw.rectangle((260,245,360,345),fill='#1c57dc')
image.save(target/'shapes.jpg',quality=95)
image=Image.new('RGB',(448,448),'white')
draw=ImageDraw.Draw(image)
for row,text in enumerate(['ORANGE SHOP','Order: A17','Apples: $3','Bread: $5','Total: $8']):
    draw.text((35,35+row*72),text,font=font,fill='black')
image.save(target/'receipt.jpg',quality=95)
print(target)
