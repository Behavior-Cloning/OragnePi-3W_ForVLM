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
table_va,n,table_name=sym('CONV2D_supported_io_types.');a=table_va;ra,rn,rname=sym('CONV2D_REG.');reg=struct.unpack_from('<4IQ',raw,offset(ra));assert reg[:3]==(2,0,8) and reg[3]*8==n and reg[4]==a
values=struct.unpack_from('<'+str(n//4)+'I',raw,offset(a));pairs=list(zip(values[::2],values[1::2]));assert len(pairs)==76
vx=(ROOT/'include/VX/vx_types.h').read_text();qh=(ROOT/'include/ovxlib/vsi_nn_tensor.h').read_text();ch=(ROOT/'include/ovxlib/utils/vsi_nn_constraint_check.h').read_text()
dtypes={int(v,16):k for k,v in re.findall(r'VX_TYPE_(INT8|INT16|INT32|INT64|UINT8|UINT16|FLOAT16|FLOAT32)\s*=\s*(0x[0-9A-Fa-f]+)',vx)}
qtypes={int(v,16):k for k,v in re.findall(r'VSI_NN_QNT_TYPE_(\w+)\s*=\s*(0x[0-9A-Fa-f]+)',qh)}
def desc(v):
 # Only low-valued actual dtypes used for this experiment are decoded here.
 return {'raw':v,'dtype':dtypes.get(v&255,'other'),'quantizer':qtypes.get(v>>8,'other')}
relevant=[{'activation':desc(a),'weight':desc(w)} for a,w in pairs if (a&255) in [2,3,4] and (w&255) in [2,3,4] and a<2048 and w<2048]
checks={name:{'pair':pair,'registered':tuple(pair) in pairs} for name,pair in {'original_A16DFP_W8DFP':[260,258],'original_A16DFP_W8PC':[260,770],'A16ASYM_W8ASYM':[516,514],'A16ASYM_W8PC':[516,770],'A8DFP_W8DFP':[258,258],'A8DFP_W8PC':[258,770],'A16DFP_W16DFP':[260,260],'A16DFP_W16PC':[260,772]}.items()}
out={'library':str(LIB),'library_sha256':hashlib.sha256(raw).hexdigest(),'table_symbol':table_name,'table_virtual_address':hex(table_va),'table_file_offset':offset(table_va),'table_bytes':n,'table_sha256':hashlib.sha256(raw[offset(table_va):offset(table_va)+n]).hexdigest(),'register':{'input_count':reg[0],'output_count':reg[1],'row_bytes':reg[2],'rows':reg[3]},'checks':checks,'relevant_rows':relevant,'all_raw_pairs':pairs,'output_dtype_claim':'This installed CONV2D op_check table registers zero output types. It does not certify output/backend support. QuantCheck and subsequent backend validation still apply.','header_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'include/VX/vx_types.h',ROOT/'include/ovxlib/vsi_nn_tensor.h',ROOT/'include/ovxlib/utils/vsi_nn_constraint_check.h']}}
# Confirm call uses input_count=2/output_count=0 and saves relevant disassembly as primary evidence.
out['op_check_disassembly']=subprocess.run(['objdump','-d','--start-address=0x88ff0','--stop-address=0x890b5',str(LIB)],check=True,stdout=subprocess.PIPE,text=True).stdout
print(json.dumps(out,indent=2))
