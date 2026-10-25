#!/usr/bin/env python3
"""Post-compile read-only qualification; fixes verifier fusion-width argument without recompiling."""
from pathlib import Path
import ast,hashlib,json,re,sys,time,numpy as np
R=Path(__file__).resolve().parent;G=R.parent;TASK=G.parent;REPO=TASK.parents[1];SRC=G/'cases/c512-m32-l0';OUT=R/'fc-axis3-c512-m32-l0'
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
# Reuse exact source constant/signature readers; all comparisons below are explicit.
tree=ast.parse((R/'build_fc_axis3.py').read_text());names={'sha','block_for','quant_sig','constant'};defs=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]);exec(compile(defs,'constant-readers','exec'))
state=json.loads((R/'initial-audit-failure.json').read_text());source_contract=verify_contract(SRC,2);candidate_contract=verify_contract(OUT,2);assert source_contract==candidate_contract
oldmeta=json.loads((SRC/'export_nbg_unify/nbg_meta.json').read_text());newmeta=json.loads((OUT/'export_nbg_unify/nbg_meta.json').read_text());assert oldmeta==newmeta
assert list(oldmeta['Inputs'])==list(newmeta['Inputs']) and list(oldmeta['Outputs'])==list(newmeta['Outputs'])
g=json.loads((SRC/'decoder.json').read_text());ng=json.loads((OUT/'decoder.json').read_text());keys=[k for k,v in g['Layers'].items() if v['op']=='fullconnect'];assert len(keys)==14
weights=[];biases=[];acts=[];oldc=(SRC/'export/vnn_decoder.c').read_text();newc=(OUT/'export/vnn_decoder.c').read_text()
for k in keys:
 assert ng['Layers'][k]['op']=='fullconnect'
 for f,rows,dtype in [('weight',weights,'INT8'),('bias',biases,'INT64')]:
  a,qa,axa,da=constant(SRC,k+':'+f);b,qb,axb,db=constant(OUT,k+':'+f);assert a==b and qa==qb and axa==axb and da==db and qa['dtype']==dtype,(k,f)
  rows.append({'key':k,'dtype':dtype,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source_exported_weight_exact':True if f=='weight' else None,'source_exported_bias_exact':True if f=='bias' else None,'quantization':qb,'channel_axis':axb,'tensor_shape':db})
 for port in [k+':out0',g['Layers'][k]['inputs'][0][1:]]:
  a=block_for(oldc,port);b=block_for(newc,port);qa=quant_sig(oldc,a);qb=quant_sig(newc,b);ta=re.search(r'VSI_NN_TYPE_(\w+)\)',a)[1];tb=re.search(r'VSI_NN_TYPE_(\w+)\)',b)[1];assert qa==qb and ta==tb=='INT16';acts.append({'port':port,'dtype':'INT16','source_quantization_exact':True,'quantization':qb})
for name in ['decoder.data','decoder.quantize','tensor-inputmeta.yml','export/decoder.export.data']:assert sha(SRC/name)==sha(OUT/name),name
ops=lambda c:re.findall(r'NEW_VXNODE\(node\[\d+\], VSI_NN_OP_(\w+),',c)
counts=lambda c:{k:ops(c).count(k) for k in sorted(set(ops(c)))}
nb=OUT/'export_nbg_unify/network_binary.nb'
manifest={'state':'complete','scope':'compiled and statically audited single block; no board validation','first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':32,'nb_path':str(nb),'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'source_graph_path':str(SRC),'source_nb_path':str(SRC/'export_nbg_unify/network_binary.nb'),'source_nb_sha256':sha(SRC/'export_nbg_unify/network_binary.nb'),'all_external_ports_exact':True,'quantizer_byte_equal':True,'source_data_byte_equal':True,'whole_exported_constant_payload_byte_equal':True,'whole_exported_constant_payload_sha256':sha(OUT/'export/decoder.export.data'),'all_14_exported_fc_weight_dtype_and_bytes_same':weights,'all_14_exported_bias_dtype_bytes_and_quantization_same':biases,'all_28_fc_input_output_dtype_and_quantization_same':acts,'actual_lowering_before':counts(oldc),'actual_lowering_after':counts(newc),'contract':candidate_contract,'nbg_meta':newmeta,'canonical_input_path':state['input'],'canonical_input_sha256':state['input_sha256'],'canonical_reference_path':state['reference'],'canonical_reference_sha256':state['reference_sha256'],'fp32_equivalence':state['graph_equivalence'],'source_hashes':state['source_hashes'],'candidate_hashes':{n:sha(OUT/n) for n in ['decoder.json','decoder.data','decoder.quantize','export/vnn_decoder.c','export/decoder.export.data','export_nbg_unify/network_binary.nb']},'initial_audit_failure':'verify_contract second argument was incorrectly first_layer=0; fixed to fusion_width=2. Original/candidate actual order equal; no NB recompile or port rewrite. initial-audit-failure.json and initial generator preserved.','compile_seconds':None,'compile_seconds_note':'Initial generator did not persist SDK duration before verifier exception; do not infer compiler duration from total script wall time.','board_used':False,'audit_time':time.time(),'audit_script_sha256':sha(__file__)}
(R/'deployment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');state.update(state='complete',nb=str(nb),nb_path=str(nb),nb_sha256=manifest['nb_sha256'],nb_bytes=manifest['nb_bytes'],all_external_ports_exact=True,quantizer_byte_equal=True,manifest=str(R/'deployment-manifest.json'),initial_audit_failure_preserved=True,error=None,actual_lowering_counts=manifest['actual_lowering_after']);(R/'status.json').write_text(json.dumps(state,indent=2)+'\n')
(OUT/'conversion.json').write_text(json.dumps({'first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':32,'scope':'FC-axis3; original quantizer and raw constants exact','compile_command':state['compile_command'],'exit_code':0,'seconds':None,**candidate_contract},indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['state','nb_path','nb_sha256','nb_bytes','all_external_ports_exact','quantizer_byte_equal','actual_lowering_before','actual_lowering_after']},indent=2))
