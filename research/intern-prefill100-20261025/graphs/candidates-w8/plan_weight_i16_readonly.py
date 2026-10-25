#!/usr/bin/env python3
"""Read-only W8-code -> I16-storage planning. Builds buffers only in RAM, no graph/candidate writes or compilation."""
from pathlib import Path
import json,re,hashlib,numpy as np
R=Path(__file__).resolve().parent;S=R/'conv1x1-c512-m32-l0';code=(S/'export/vnn_decoder.c').read_text();raw=(S/'export/decoder.export.data').read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest();items=[];weights=[];pattern=r'NEW_CONST_TENSOR\(([^,]+), attr, VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)'
for names,b in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',code,re.S):
 m=re.search(pattern,b)
 if not m:continue
 names=names.split();off,n=int(m[3]),int(m[4]);weight=any(re.fullmatch('@b[01]_fullconnect_\d+:weight',x) for x in names);item={'names':names,'tensor':m[1],'dtype':m[2],'offset':off,'bytes':n,'weight':weight,'call':m[0],'data_sha256':sha(raw[off:off+n])};items.append(item)
 if weight:
  assert item['dtype']=='INT8';old=np.frombuffer(raw,dtype='i1',count=n,offset=off);new=old.astype('<i2');assert np.array_equal(old.astype(np.int64),new.astype(np.int64));assert new.astype('i1').tobytes()==raw[off:off+n]
  typ=re.search(r'attr.dtype.qnt_type = (\w+);',b)[1];fl=re.search(r'attr.dtype.fl = (-?\d+);',b);scale_id=re.search(r'attr.dtype.scales = (\w+);',b)
  if fl:scale=np.float32(2.**(-int(fl[1])));f_old=old.astype(np.float32)*scale;f_new=new.astype(np.float32)*scale;q={'qnt_type':typ,'fl':int(fl[1]),'scale_f32':float(scale)}
  else:
   assert scale_id;sm=re.search(r'\b'+scale_id[1]+r'\[\]\s*=\s*\{([^}]+)\}',code);scale=np.asarray([float(x.strip().rstrip('f')) for x in sm[1].split(',') if x.strip()],dtype='<f4');o=len(scale);f_old=old.reshape(o,-1).astype(np.float32)*scale[:,None];f_new=new.reshape(o,-1).astype(np.float32)*scale[:,None];q={'qnt_type':typ,'scales_sha256':sha(scale.tobytes()),'scales_count':o,'channel_axis':int(re.search(r'attr.dtype.channel_dim = (\d+);',b)[1])}
  assert f_old.tobytes()==f_new.tobytes();weights.append({**item,'new_dtype':'INT16','expanded_bytes':new.nbytes,'expanded_sha256':sha(new.tobytes()),'integer_range':[int(old.min()),int(old.max())],'recovered_i8_sha256':sha(new.astype('i1').tobytes()),'dequant_f32_bitwise_same':True,'original_quantization_preserved':q})
assert len(weights)==14 and len(items)==32
unique=sorted({(x['offset'],x['bytes']) for x in items});cursor=0
for off,n in unique:assert off==cursor,(off,cursor);cursor+=n
assert cursor==len(raw),'unreferenced gap/header would require explicit treatment'
# Compact exact-payload rebuild, simulated only in memory. Aliased old ranges stay aliased.
wmap={(x['offset'],x['bytes']):x for x in weights};buf=bytearray();mapping={}
for off,n in unique:
 oldbytes=raw[off:off+n];newbytes=np.frombuffer(oldbytes,'i1').astype('<i2').tobytes() if (off,n) in wmap else oldbytes;mapping[(off,n)]=(len(buf),len(newbytes));buf.extend(newbytes)
newcode=code;refs=[]
for x in items:
 no,nn=mapping[(x['offset'],x['bytes'])];nt='INT16' if x['weight'] else x['dtype'];replacement='NEW_CONST_TENSOR('+x['tensor']+', attr, VSI_NN_TYPE_'+nt+', '+str(no)+', '+str(nn)+')';assert newcode.count(x['call'])==1;newcode=newcode.replace(x['call'],replacement);refs.append({'names':x['names'],'tensor':x['tensor'],'old_dtype':x['dtype'],'new_dtype':nt,'old_offset':x['offset'],'old_bytes':x['bytes'],'new_offset':no,'new_bytes':nn,'is_weight':x['weight']})
for x in items:
 no,nn=mapping[(x['offset'],x['bytes'])];actual=bytes(buf[no:no+nn])
 if x['weight']:assert np.frombuffer(actual,'<i2').astype('i1').tobytes()==raw[x['offset']:x['offset']+x['bytes']]
 else:assert actual==raw[x['offset']:x['offset']+x['bytes']]
# Removing constant declarations makes source/target C identical: all nodes, tensor grids and ports unchanged.
def without_calls(s):return re.sub(pattern,'CONST_READ_PLACEHOLDER',s)
assert without_calls(newcode)==without_calls(code)
weightbytes=sum(x['bytes'] for x in weights)
out={'state':'read_only_feasibility_complete','candidate_written':False,'compiled':False,'board_used':False,'source_export':str(S/'export'),'source_c_sha256':sha(code.encode()),'source_payload_sha256':sha(raw),'source_payload_bytes':len(raw),'weight_tensors':14,'weight_i8_bytes':weightbytes,'weight_i16_bytes':weightbytes*2,'unique_other_constant_bytes':len(raw)-weightbytes,'compact_rebuild_payload_bytes':len(buf),'compact_increase_bytes':len(buf)-len(raw),'compact_rebuild_payload_sha256':sha(buf),'compact_simulated_c_sha256':sha(newcode.encode()),'append_only_payload_bytes_without_alignment':len(raw)+weightbytes*2,'append_only_increase_bytes':weightbytes*2,'all_weights_dequant_f32_bitwise_same':True,'all_nonweight_constant_bytes_same':True,'all_quantizer_attributes_same':True,'all_graph_nodes_and_external_ports_unchanged':True,'source_constant_allocations':32,'unique_source_segments':len(unique),'alias_note':'Four distinct I64 zero-bias tensor declarations share original (offset6,size7168), retain alias; one grouped bias declaration feeds four FCs.','weights':weights,'compact_constant_reference_changes':refs,'limits':'dtype changes I8 to I16 storage; not original bytes. Original integer code and dequantized values exact. SDK type check supports A16DFP x W16DFP/PC but VerifyGraph/backend/kernel/latency untested. Never requantize original FP32 source.'}
(R/'weight-i16-feasibility.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['weights','compact_constant_reference_changes']},indent=2))
