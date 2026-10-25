from pathlib import Path
import subprocess,json,re,hashlib,struct
r=Path('/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator');p=r/'lib/libovxlib.so';raw=p.read_bytes()
e=struct.unpack_from('<16sHHIQQQIHHHHHH',raw,0);ss=[struct.unpack_from('<IIQQQQIIQQ',raw,e[6]+i*e[11]) for i in range(e[12])]
def off(va):
 for x in ss:
  if x[3]<=va<x[3]+x[5] and x[1]!=8:return x[4]+va-x[3]
 raise ValueError(hex(va))
symbols=subprocess.run(['readelf','-Ws',str(p)],text=True,capture_output=True,check=True).stdout
syms=[x.split() for x in symbols.splitlines() if re.match(r'^\s*\d+:\s+[0-9a-f]+\s+\d+',x) and len(x.split())>=8]
refs={int(x[1],16):x[-1] for x in syms if x[1]!='0000000000000000'}
obj=struct.unpack_from('<7Q',raw,off(0x7d92a0))
o={'library_sha256':hashlib.sha256(raw).hexdigest(),'object_words':[hex(x) for x in obj],'object_names':[refs.get(x) for x in obj], 'header':(r/'include/ovxlib/ops/vsi_nn_op_bitcast.h').read_text(),'definitions':{}}
for p2 in (r/'include').rglob('*.h'):
 t=p2.read_text(errors='replace')
 if 'vxBitCastLayer' in t:
  pos=t.index('vxBitCastLayer');o['definitions'][str(p2)]=t[max(0,pos-1800):pos+900]
for va in obj:
 if va in refs and va<0x500000:
  size=next((int(x[2]) for x in syms if int(x[1],16)==va),256)
  o.setdefault('disassembly',{})[hex(va)]=subprocess.run(['objdump','-d','--start-address='+hex(va),'--stop-address='+hex(va+min(size,2048)),str(p)],text=True,capture_output=True,check=True).stdout
for va,n in [(0x1b2470,23),(0xba8f0,10)]:o.setdefault('disassembly',{})[hex(va)]=subprocess.run(['objdump','-d','--start-address='+hex(va),'--stop-address='+hex(va+n),str(p)],text=True,capture_output=True,check=True).stdout
print(json.dumps(o))
