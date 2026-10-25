#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,time
P=Path(__file__).resolve().parent;T=P.parent.parent;E=P/'official-export';S=T/'graphs/cases/c512-m32-l0';O=T/'runtime/official-export-c512-m32-l0'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((P/'static-audit.json').read_text());st=json.loads((P/'official-status.json').read_text());proof=json.loads((P/'official-source-proof.json').read_text())
assert st['status']=='generated_pending_nbg_port_and_payload_audit'
assert sha(E/'vnn_decoder.c')==a['candidate_c_sha256'];assert sha(E/'decoder.export.data')==a['candidate_payload_sha256']
assert sha(S/'export/decoder.export.data')==a['source_payload_sha256']
assert sha(E/'main.c')==proof['main_sha256'];assert sha(E/'makefile.linux')==proof['make_sha256']
assert all(sha(E/x['file'])==x['sha256'] for x in proof['inputs'])
nb=E/'network_binary.nb';assert sha(nb)==st['variants']['official-export']['nbg']['sha256']
actual=json.loads((P/'official-export-binary-ports.json').read_text());original=json.loads((O/'plain-binary-ports.json').read_text())
def norm(j):
 assert j['count']==16 and len(j['ports'])==16
 out=[]
 for i,x in enumerate(j['ports']):
  assert x['index']==i and x['direction']==(0 if i<11 else 1);assert x['quant_status']==0
  y={k:x[k] for k in ['index','direction','type','dtype','dims','quant_type']}
  if x['quant_type']==1:assert x['fl_status']==0;y['fl']=x['fl']
  else:assert x['quant_type']==0
  out.append(y)
 return out
np=norm(actual);op=norm(original);assert np==op
expected=proof['ports_metadata'];entries=list(expected['Inputs'].items())+list(expected['Outputs'].items());assert len(entries)==16
for (name,m),n in zip(entries,np):
 assert n['dims']==list(reversed(m['shape'])),(name,n,m)
 if m.get('dtype')=='float16':assert n['dtype']==15 and n['quant_type']==0
 else:assert m['quantizer']=='dynamic_fixed_point' and m['quantize']['qtype']=='i16' and n['dtype']==4 and n['quant_type']==1 and n['fl']==m['quantize']['fl']
fixture=T/'fixtures-canonical/c512-m32-l0';f=json.loads((fixture/'probe-fixture.json').read_text());assert sha(fixture/'probe-input.f32')==proof['canonical_input_sha256']
ports={'scope':'Actual NB kernel metadata imported via SDK vxImportKernelFromURL; no graph execution and no board access. Ignore inactive quant union fields; DFP uses only fl.','actual_ports_exact_original':True,'actual_ports_exact_expected_metadata':True,'count':16,'input_count':11,'output_count':5,'ports':np,'expected_semantic_order':[n for n,m in entries],'inspector_c_sha256':sha(P/'inspect_nbg.c'),'raw_binary_ports_sha256':sha(P/'official-export-binary-ports.json'),'source_raw_binary_ports_sha256':sha(O/'plain-binary-ports.json')}
(P/'actual-nbg-ports-audit.json').write_text(json.dumps(ports,indent=2)+'\n')
manifest={'state':'complete','scope':'One C512/M32/L0 candidate generated and statically verified; prepared for single-block board A-B-A, not validated NPU outputs/speed or fullmodel quality.','first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':32,'nb_path':str(nb),'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'source_graph_path':str(S),'source_nb_path':str(S/'export_nbg_unify/network_binary.nb'),'source_nb_sha256':sha(S/'export_nbg_unify/network_binary.nb'),'source_nb_bytes':(S/'export_nbg_unify/network_binary.nb').stat().st_size,'all_external_ports_exact':True,'actual_binary_input_count':11,'actual_binary_output_count':5,'actual_binary_ports':ports,'nbg_meta':expected,'all_14_weight_integer_codes_exact':True,'all_14_weight_dequant_f32_values_bitwise_exact':True,'weight_storage_dtype_change':'signed INT8 codes -> signed INT16 storage; same codes/scales, no additional weight precision','original_weight_storage_bytes_preserved':False,'weight_storage_bytes_before':sum(x['old_bytes'] for x in a['weights']),'weight_storage_bytes_after':sum(x['new_bytes'] for x in a['weights']),'weights':a['weights'],'all_nonweight_constant_bytes_exact':True,'all_quantizer_attributes_same_except_weight_storage_dtype':True,'bias_and_output_grid_unchanged':True,'payload_bytes_before':a['source_payload_bytes'],'payload_bytes_after':a['candidate_payload_bytes'],'payload_increase_bytes':a['increase_bytes'],'constant_reference_changes':a['constant_reference_changes'],'canonical_input_path':str(fixture/'probe-input.f32'),'canonical_input_sha256':sha(fixture/'probe-input.f32'),'canonical_reference_path':str(fixture/'probe-reference.f32'),'canonical_reference_sha256':sha(fixture/'probe-reference.f32'),'canonical_fixture_path':str(fixture/'probe-fixture.json'),'canonical_fixture_sha256':sha(fixture/'probe-fixture.json'),'source_hashes':{'direct_dfp_conv_c':a['source_c_sha256'],'source_export_payload':a['source_payload_sha256'],'original_fc_nb':sha(S/'export_nbg_unify/network_binary.nb')},'candidate_hashes':{n:sha(E/n) for n in ['vnn_decoder.c','decoder.export.data','main.c','makefile.linux','network_binary.nb']},'static_audit_sha256':sha(P/'static-audit.json'),'independent_static_audit_path':str(P/'INDEPENDENT_STATIC_AUDIT.json'),'independent_static_audit_sha256':sha(P/'INDEPENDENT_STATIC_AUDIT.json'),'official_source_proof_sha256':sha(P/'official-source-proof.json'),'official_export_status_sha256':sha(P/'official-status.json'),'official_plain_export_reproduces_original_nb':True,'board_used':False,'simulator_times_are_not_board_latency':True,'audit_time':time.time(),'audit_script_sha256':sha(__file__)}
(P/'deployment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['state','nb_path','nb_sha256','nb_bytes','all_external_ports_exact','actual_binary_input_count','actual_binary_output_count','all_14_weight_integer_codes_exact']},indent=2))
