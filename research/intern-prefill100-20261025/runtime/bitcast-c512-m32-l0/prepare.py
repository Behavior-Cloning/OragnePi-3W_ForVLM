#!/usr/bin/env python3
"""One independent generated-C BITCAST attempt. No model/port/grid changes."""
from pathlib import Path
import hashlib,json,re,shutil,difflib,struct
P=Path(__file__).resolve().parent;T=P.parent.parent
S=T/'graphs/candidates-w8/conv1x1-affine-c512-m32-l0/export';E=P/'export'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not E.exists();shutil.copytree(S,E,ignore=shutil.ignore_patterns('*.o','gen_nbg'))
c=(E/'vnn_decoder.c').read_text();assert sha(S/'vnn_decoder.c')=='0d7a158d4a2da4ca114e5b91db1d04a4ca412105ae4507d7137c54d389389701'
attrs={}
pat=r'memset\(\s*&attr,\s*0,\s*sizeof\(\s*attr\s*\)\s*\);(?P<body>[^{}]*?)NEW_VIRTUAL_TENSOR\(node\[(?P<node>\d+)\]->output.tensors\[(?P<port>\d+)\],\s*attr,\s*(?P<dtype>\w+)\);'
for m in re.finditer(pat,c,re.S):attrs[int(m['node']),int(m['port'])]={'dtype':m['dtype'],'assignments':dict(re.findall(r'attr.dtype.(\w+)\s*=\s*([^;]+);',m['body']))}
pat=r'NEW_VXNODE\(node\[(\d+)\],\s*VSI_NN_OP_DATACONVERT,\s*1,\s*1,\s*(\d+)\);'
nodes=list(re.finditer(pat,c));assert len(nodes)==14
expected=json.loads((T/'runtime/bitcast-review/generated-c-converters.json').read_text())['converters'];proof=[]
for m in nodes:
 n,uid=map(int,m.groups());im=re.search(r'node\['+str(n)+r'\]->input.tensors\[0\]\s*=\s*node\[(\d+)\]->output.tensors\[(\d+)\];',c);assert im
 src=tuple(map(int,im.groups()));ia=attrs[src];oa=attrs[n,0]
 assert ia['dtype']==oa['dtype']=='VSI_NN_TYPE_INT16'
 assert ia['assignments']['qnt_type']=='VSI_NN_QNT_TYPE_DFP'
 assert oa['assignments']['qnt_type']=='VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC'
 fl=int(ia['assignments']['fl']);scale=float(oa['assignments']['scale']);zp=int(oa['assignments']['zero_point'])
 assert zp==0 and scale==2.0**(-fl)
 assert all(i*2.0**(-fl)==(i-zp)*scale for i in range(-32768,32768))
 prior=next(x for x in expected if x['node_index']==n);assert prior['uid']==uid and prior['input']==ia and prior['output']==oa
 proof.append({'node':n,'uid':uid,'source_node':src[0],'source_port':src[1],'dtype':'signedI16','source_quant':'DFP','destination_quant':'AFFINE_ASYMMETRIC','fl':fl,'scale':scale,'zero_point':zp,'all65536_integer_values_equal':True})
patched=re.sub(pat,lambda m:m[0].replace('VSI_NN_OP_DATACONVERT','VSI_NN_OP_BITCAST'),c)
assert patched.replace('VSI_NN_OP_BITCAST','VSI_NN_OP_DATACONVERT')==c and patched.count('VSI_NN_OP_BITCAST')==14
(E/'vnn_decoder.c').write_text(patched)
(P/'decoder.patch').write_text(''.join(difflib.unified_diff(c.splitlines(True),patched.splitlines(True),fromfile='original/vnn_decoder.c',tofile='bitcast/vnn_decoder.c')))
manifest={x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in S.iterdir() if x.is_file() and x.name!='gen_nbg' and x.suffix!='.o'}
for name in manifest:
 if name!='vnn_decoder.c':assert sha(E/name)==manifest[name]['sha256']
(P/'patch-proof.json').write_text(json.dumps({'source':str(S),'source_files':manifest,'decoder_sha256':sha(E/'vnn_decoder.c'),'converters':proof,'scope':'Only14 node opcode substitutions; attrs,weightdata,graph ports unchanged. Main is separately replaced to verify and export only.'},indent=2)+'\n')
print('prepared14bitcast',sha(E/'vnn_decoder.c'))
