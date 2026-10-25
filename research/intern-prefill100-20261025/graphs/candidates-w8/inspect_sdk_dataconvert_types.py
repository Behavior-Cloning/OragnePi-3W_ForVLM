#!/usr/bin/env python3
"""Read installed SDK ELF support table; no graph compilation or device access."""
from pathlib import Path
import subprocess,struct,re,json,hashlib
ROOT=Path('/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator');LIB=ROOT/'lib/libovxlib.so';raw=LIB.read_bytes()
symbols=subprocess.run(['readelf','-Ws',str(LIB)],check=True,stdout=subprocess.PIPE,text=True).stdout
# ELF64 section mapping from file itself; no hard-coded virtual-to-file delta.
e=struct.unpack_from('<16sHHIQQQIHHHHHH',raw,0);assert e[0][:5]==b'\x7fELF\x02';shoff=e[6];shentsize=e[11];shnum=e[12]
sections=[struct.unpack_from('<IIQQQQIIQQ',raw,shoff+i*shentsize) for i in range(shnum)]
def offset(va):
 for x in sections:
  if x[3]<=va<x[3]+x[5] and x[1]!=8:return x[4]+va-x[3]
 raise AssertionError(hex(va))
def sym(name):
 rows=[line.split() for line in symbols.splitlines() if name in line]
 rows=[x for x in rows if x[-1].startswith(name)]
 assert len(rows)==1,(name,rows)
 return int(rows[0][1],16),int(rows[0][2]),rows[0][-1]

a,n,name=sym('DATACONVERT_supported_io_types.');ra,rn,rname=sym('DATACONVERT_REG.');reg=struct.unpack_from('<4IQ',raw,offset(ra));v=struct.unpack_from('<'+str(n//4)+'I',raw,offset(a));pairs=list(zip(v[::2],v[1::2]));assert reg[:3]==(1,1,8)
checks={name:{'pair':p,'registered':tuple(p) in pairs} for name,p in {'I16DFP_to_I16ASYM':[260,516],'I16ASYM_to_I16DFP':[516,260],'I16DFP_to_FP32':[260,10],'FP32_to_I16ASYM':[10,516],'I16ASYM_to_FP32':[516,10],'FP32_to_I16DFP':[10,260],'I16DFP_to_I8DFP':[260,258],'I8DFP_to_I16DFP':[258,260]}.items()}
print(json.dumps({'library_sha256':hashlib.sha256(raw).hexdigest(),'table':name,'register':list(reg[:4]),'checks':checks,'all_raw_pairs':pairs},indent=2))
