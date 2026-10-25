#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,re,shutil,difflib
P=Path(__file__).resolve().parent;T=P.parent.parent;SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--affine-export',type=Path,required=True);args=a.parse_args();A=args.affine_export;S=T/'graphs/cases/c512-m256-l0/export';old=T/'runtime/official-export-c512-m32-l0'
assert not (P/'plain').exists() and not (P/'bitcast').exists()
assert SHA(S/'decoder.export.data')==SHA(A/'decoder.export.data')
for v,src in [('plain',S),('bitcast',A)]:
 shutil.copytree(src,P/v,ignore=shutil.ignore_patterns('*.o','gen_nbg','*.nb','ports-*.json'))
 shutil.copy2(S/'main.c',P/v/'main.c');shutil.copy2(old/'plain/makefile.linux',P/v/'makefile.linux')
c=(P/'bitcast/vnn_decoder.c').read_text();assert 'VSI_NN_OP_BITCAST' not in c
attrs={}
pat=r'memset\(\s*&attr,\s*0,\s*sizeof\(\s*attr\s*\)\s*\);(?P<body>[^{}]*?)NEW_VIRTUAL_TENSOR\(node\[(?P<node>\d+)\]->output.tensors\[(?P<port>\d+)\],\s*attr,\s*(?P<dtype>\w+)\);'
for m in re.finditer(pat,c,re.S):attrs[int(m['node']),int(m['port'])]={'dtype':m['dtype'],'assignments':dict(re.findall(r'attr.dtype.(\w+)\s*=\s*([^;]+);',m['body']))}
pat=r'NEW_VXNODE\(node\[(\d+)\],\s*VSI_NN_OP_DATACONVERT,\s*1,\s*1,\s*(\d+)\);';nodes=list(re.finditer(pat,c));assert len(nodes)==14;converters=[]
for m in nodes:
 n,uid=map(int,m.groups());im=re.search(r'node\['+str(n)+r'\]->input.tensors\[0\]\s*=\s*node\[(\d+)\]->output.tensors\[(\d+)\];',c);assert im
 src=tuple(map(int,im.groups()));ia=attrs[src];oa=attrs[n,0]
 assert ia['dtype']==oa['dtype']=='VSI_NN_TYPE_INT16' and ia['assignments']['qnt_type']=='VSI_NN_QNT_TYPE_DFP' and oa['assignments']['qnt_type']=='VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC'
 fl=int(ia['assignments']['fl']);scale=float(oa['assignments']['scale']);zp=int(oa['assignments']['zero_point']);assert zp==0 and scale==2.0**(-fl)
 assert all(i*2.0**(-fl)==(i-zp)*scale for i in range(-32768,32768))
 converters.append({'node_index':n,'uid':uid,'input_node':src[0],'input_port':src[1],'input':ia,'output':oa,'all65536_integer_values_equal':True})
patched=re.sub(pat,lambda m:m[0].replace('VSI_NN_OP_DATACONVERT','VSI_NN_OP_BITCAST'),c);assert patched.replace('VSI_NN_OP_BITCAST','VSI_NN_OP_DATACONVERT')==c
(P/'bitcast/vnn_decoder.c').write_text(patched)
(P/'decoder.patch').write_text(''.join(difflib.unified_diff(c.splitlines(True),patched.splitlines(True),fromfile='affine/vnn_decoder.c',tofile='bitcast/vnn_decoder.c')))
(P/'patch-proof.json').write_text(json.dumps({'source_affine_export':str(A),'original_affine_c_sha256':SHA(A/'vnn_decoder.c'),'patched_c_sha256':SHA(P/'bitcast/vnn_decoder.c'),'converters':converters,'scope':'only14 converter opcode changes from generated affine C; constants payload and other tensor metadata untouched'},indent=2)+'\n')
meta=json.loads((S.parent/'export_nbg_unify/nbg_meta.json').read_text());order=json.loads((S.parent/'conversion.json').read_text())['input_order'];assert order==list(meta['Inputs'])
inputs=[]
for i,name in enumerate(order):
 shape=meta['Inputs'][name]['shape'];n=1
 for v in shape:n*=v
 inputs.append({'index':i,'name':name,'file':name+'_0.tensor','elements':n,'shape':shape})
proof={'scope':'C512M256L0 only; official main,14 BITCAST equivalent I16 conversions; no board','source_original':str(S),'source_affine':str(A),'main_sha256':SHA(P/'plain/main.c'),'make_sha256':SHA(P/'plain/makefile.linux'),'inputs':inputs,'ports_metadata':meta,'environment_evidence':'M32 plain exact reproducer and runtime/bitcast-export-audit official SDK casegenerator disassembly','variants':{v:{'c_sha256':SHA(P/v/'vnn_decoder.c'),'data_sha256':SHA(P/v/'decoder.export.data')} for v in ['plain','bitcast']}}
(P/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n');(P/'nbg_meta.expected.json').write_text(json.dumps(meta,indent=2)+'\n')
assert SHA(P/'plain/main.c')==SHA(P/'bitcast/main.c')
print('prepared M256,14exactgrid,sourceweights unchanged')
