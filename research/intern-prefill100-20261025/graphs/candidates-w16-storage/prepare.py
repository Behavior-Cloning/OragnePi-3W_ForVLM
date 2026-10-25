#!/usr/bin/env python3
"""Build exactly one W8-code-in-I16-storage generated-C candidate, no compilation."""
from pathlib import Path
import hashlib,json,re,shutil,difflib,numpy as np
P=Path(__file__).resolve().parent;T=P.parent.parent
S=T/'graphs/candidates-w8/conv1x1-c512-m32-l0/export';E=P/'export'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not E.exists(),'preserve prior evidence; never overwrite'
code=(S/'vnn_decoder.c').read_text();raw=(S/'decoder.export.data').read_bytes()
assert sha(code.encode())=='10a772e70f4ecc8300129f59ae4717e67a4f7def316be4fdfc8007607e65f8e3'
assert sha(raw)=='839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2'
pattern=r'NEW_CONST_TENSOR\(([^,]+), attr, VSI_NN_TYPE_(\w+),\s*(\d+),\s*(\d+)\)'
items=[];weight_proof=[]
for names,b in re.findall(r'/\* (.*?)\*/(.*?)(?=/\*|$)',code,re.S):
 m=re.search(pattern,b)
 if not m:continue
 names=names.split();off,n=int(m[3]),int(m[4]);weight=any(re.fullmatch('@b[01]_fullconnect_\d+:weight',x) for x in names)
 item={'names':names,'tensor':m[1],'dtype':m[2],'old_offset':off,'old_bytes':n,'weight':weight,'call':m[0],'source_bytes_sha256':sha(raw[off:off+n])};items.append(item)
 if not weight:continue
 assert item['dtype']=='INT8';old=np.frombuffer(raw,'i1',count=n,offset=off);new=old.astype('<i2')
 assert np.array_equal(old.astype(np.int64),new.astype(np.int64));assert new.astype('i1').tobytes()==raw[off:off+n]
 typ=re.search(r'attr.dtype.qnt_type = (\w+);',b)[1];fl=re.search(r'attr.dtype.fl = (-?\d+);',b);scale_id=re.search(r'attr.dtype.scales = (\w+);',b)
 if fl:
  scale=np.float32(2.**(-int(fl[1])));a=old.astype(np.float32)*scale;c=new.astype(np.float32)*scale
  q={'qnt_type':typ,'fl':int(fl[1]),'scale_f32':float(scale)}
 else:
  assert scale_id;sm=re.search(r'\b'+scale_id[1]+r'\[\]\s*=\s*\{([^}]+)\}',code)
  scale=np.asarray([float(x.strip().rstrip('f')) for x in sm[1].split(',') if x.strip()],dtype='<f4')
  a=old.reshape(len(scale),-1).astype(np.float32)*scale[:,None];c=new.reshape(len(scale),-1).astype(np.float32)*scale[:,None]
  q={'qnt_type':typ,'scales_sha256':sha(scale.tobytes()),'scales_count':len(scale),'channel_axis':int(re.search(r'attr.dtype.channel_dim = (\d+);',b)[1])}
 assert a.tobytes()==c.tobytes()
 weight_proof.append({'names':names,'source_bytes_sha256':item['source_bytes_sha256'],'candidate_bytes_sha256':sha(new.tobytes()),'recovered_i8_bytes_sha256':sha(new.astype('i1').tobytes()),'old_dtype':'INT8','new_dtype':'INT16','old_bytes':n,'new_bytes':new.nbytes,'integer_min':int(new.min()),'integer_max':int(new.max()),'integer_values_exact':True,'dequant_f32_bitwise_exact':True,'quantization_unchanged':q})
assert len(weight_proof)==14 and len(items)==32
unique=sorted({(x['old_offset'],x['old_bytes']) for x in items});cursor=0
for off,n in unique:assert off==cursor;cursor+=n
assert cursor==len(raw)
wkeys={(x['old_offset'],x['old_bytes']) for x in items if x['weight']};buf=bytearray();mapping={}
for off,n in unique:
 b=raw[off:off+n];b=np.frombuffer(b,'i1').astype('<i2').tobytes() if (off,n) in wkeys else b
 mapping[off,n]=(len(buf),len(b));buf.extend(b)
newcode=code
for x in items:
 no,nn=mapping[x['old_offset'],x['old_bytes']];typ='INT16' if x['weight'] else x['dtype']
 call=f"NEW_CONST_TENSOR({x['tensor']}, attr, VSI_NN_TYPE_{typ}, {no}, {nn})"
 assert newcode.count(x['call'])==1;newcode=newcode.replace(x['call'],call)
 actual=bytes(buf[no:no+nn]);src=raw[x['old_offset']:x['old_offset']+x['old_bytes']]
 assert (np.frombuffer(actual,'<i2').astype('i1').tobytes()==src) if x['weight'] else (actual==src)
 x.update(new_offset=no,new_bytes=nn,new_dtype=typ,actual_bytes_sha256=sha(actual),nonweight_bytes_exact=None if x['weight'] else True)
 del x['call']
assert re.sub(pattern,'CONST',code)==re.sub(pattern,'CONST',newcode)
new_intervals=sorted(set(mapping.values()));cursor=0
for off,n in new_intervals:assert off==cursor;cursor+=n
assert cursor==len(buf)
assert sha(buf)=='08d1a7880e6924f938853df20a00f5a760527ca757345692ca789ccfb1c1e50f'
assert sha(newcode.encode())=='e995dd7c770ad4b1f3af50ca2aaa3bb01dbe1d4f0aaa379da98dced402ae238f'
shutil.copytree(S,E,ignore=shutil.ignore_patterns('*.o','gen_nbg','decoder.export.data','network_binary.nb'))
(E/'vnn_decoder.c').write_text(newcode);(E/'decoder.export.data').write_bytes(buf)
(P/'decoder.patch').write_text(''.join(difflib.unified_diff(code.splitlines(True),newcode.splitlines(True),fromfile='source/vnn_decoder.c',tofile='w16-storage/vnn_decoder.c')))
(P/'main-original.c.source').write_text((E/'main.c').read_text())
main=(T/'runtime/bitcast-c512-m32-l0/main-export.c.source').read_text()
start=main.index(' printf("STAGE nbg_size_query');end=main.index(' done:',start)
main=main[:start]+' rc=0;\n printf("VERIFY_ONLY_NO_NBG_EXPORT\\n");\n'+main[end:]
main=main.replace('if(argc!=3)','if(argc!=2)').replace('decoder.export.data output.nb','decoder.export.data')
assert 'vsi_nn_GenerateNBG(' not in main
(E/'main.c').write_text(main);(P/'main-verify.c.source').write_text(main)
proof={'state':'prepared_static_audit_passed','source_export':str(S),'candidate_export':str(E),'source_c_sha256':sha(code.encode()),'source_payload_sha256':sha(raw),'candidate_c_sha256':sha(newcode.encode()),'candidate_payload_sha256':sha(buf),'source_payload_bytes':len(raw),'candidate_payload_bytes':len(buf),'increase_bytes':len(buf)-len(raw),'source_constant_allocations':len(items),'unique_source_segments':len(unique),'unique_candidate_segments':len(new_intervals),'all_unique_intervals_contiguous_nonoverlapping':True,'aliases_preserved':True,'all_14_integer_codes_exact':True,'all_14_dequant_f32_values_bitwise_exact':True,'all_nonweight_constant_bytes_exact':True,'all_quantizer_attributes_same_except_storage_dtype':True,'all_nodes_connections_external_ports_identical':True,'weights':weight_proof,'constant_reference_changes':items,'main_sha256':sha(main.encode()),'scope':'14 I8 storage->I16 with original codes/scales, no FP32 requantization. Verify-only main has no NBG export call. No board; compiled separately.'}
(P/'static-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k not in ['weights','constant_reference_changes']},indent=2))
