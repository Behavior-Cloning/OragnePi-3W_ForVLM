#!/usr/bin/env python3
"""Read-only artifact validation, then publish delivery JSON; never compile or run a model."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
J=lambda n:json.loads((R/n).read_text())
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
S=J('source-proof.json'); P=J('patch-proof.json'); A=J('INDEPENDENT_EXPORT_AUDIT.json'); T=J('status.json')
assert H(R/'INDEPENDENT_EXPORT_AUDIT.json')=='08c480712cc22273d30d54d617b1f3ce5fdf5f276e9639533a6954870d41bbad'
assert A['audit_pass'] and A['all_constant_integer_bytes_and_numeric_values_equal']
assert A['weights']==14 and A['biases']==14 and A['constant_count']==28 and A['qtype_changed_count']==12
assert A['all11_actual_text_inputs_reconstructed_bitwise'] and A['all16_actual_nb_ports_byte_equal']
assert A['actual_nb_ports_match_expected_geometry_and_active_quant'] and A['official_argv_and_environment_exact']
assert A['plain_reproduces_source_nb_bitwise'] and A['official_main_exact']
assert len(P['converters'])==14 and all(x['all65536_integer_values_equal'] for x in P['converters'])
assert all(x['raw_and_numeric_equal'] for x in A['constant_records'])
weights=[x for x in A['constant_records'] if x['key'].endswith(':weight')]
assert len(weights)==14 and all(x['candidate']['dtype']==x['source']['dtype']=='INT8' for x in weights)
a=J('plain-binary-ports.json'); b=J('bitcast-binary-ports.json'); expected=J('nbg_meta.expected.json')
assert a==b and a['count']==16
assert [x['direction'] for x in a['ports']]==[0]*11+[1]*5
norm=[]
for i,(port,(name,meta)) in enumerate(zip(b['ports'],list(expected['Inputs'].items())+list(expected['Outputs'].items()))):
 assert port['index']==i and port['type']==2069 and port['dims']==meta['shape'][::-1]
 assert all(port[k]==0 for k in ['quant_status','fl_status','zp_status','scale_status'])
 if 'quantizer' in meta:
  assert meta['quantizer']=='dynamic_fixed_point' and port['quant_type']==1 and port['dtype']==4 and port['fl']==meta['quantize']['fl']
  dtype='int16'; quant={'type':'DFP','fl':port['fl'],'scale':2.0**(-port['fl']),'zero_point':0}
 else:
  assert meta['dtype']=='float16' and port['quant_type']==0 and port['dtype']==15
  dtype='float16'; quant={'type':'none'}
 norm.append({'index':i,'direction':'input' if i<11 else 'output','source_name':name,'dimensions_vx':port['dims'],'dtype':dtype,'quantization':quant})
assert norm[0]['dimensions_vx']==[896,256,1]
for label,sha,size in [('plain','e9b2d8af40f96341eda1e7ff07b867ccde9bbcf9a755746c86e6dafec2de11d5',35976976),('bitcast','907689cdee9a1f767f4b64d8854bcbae27b5c16871f26500519ed977fee4c250',35975168)]:
 p=R/label/'network_binary.nb'; assert H(p)==sha and p.stat().st_size==size
 assert T['variants'][label]['make']['rc']==T['variants'][label]['generate']['rc']==0
 assert T['variants'][label]['nbg']['sha256']==sha
assert S['source_nb_sha256']==H(R/'plain/network_binary.nb')
input_path=Path(S['input_sources']); fixture_path=input_path.with_name('probe-fixture.json'); F=json.loads(fixture_path.read_text())
assert H(input_path)==S['canonical_input_sha256']==F['input_sha256']
assert input_path.stat().st_size==A['canonical_f32_bytes']==13238272
assert F['capacity']==512 and F['query_tokens']==256 and F['history']==128 and F['first_layer']==0 and F['fusion_width']==2
sources={n:H(R/n) for n in ['source-proof.json','patch-proof.json','prepare_bitcast.py','prepare_inputs.py','run_official.py','inspect_binary_ports.py','inspect_nbg.c','status.json','finalize_manifest.py','INDEPENDENT_EXPORT_AUDIT.json']}
m={'state':'complete','qualification':'Host export and exact actual binary-port/constant audit only; board validation is pending.',
 'geometry':{'first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':256},
 'nb_path':str(R/'bitcast/network_binary.nb'),'nb_sha256':H(R/'bitcast/network_binary.nb'),'nb_bytes':35975168,
 'source_nb_sha256':S['source_nb_sha256'],'source_nb_path':str(R/'plain/network_binary.nb'),
 'all_external_ports_exact':True,'all_14_weight_values_exact':True,'all_14_weight_integer_codes_exact':True,
 'all_14_bias_values_exact':True,'all_external_quant_grids_equal':True,'weight_storage':'i8-preserved',
 'actual_binary_ports':norm,'bitcast_opcode_changes':14,
 'constant_audit':{'path':str(R/'INDEPENDENT_EXPORT_AUDIT.json'),'sha256':H(R/'INDEPENDENT_EXPORT_AUDIT.json'),'weight_count':14,'bias_count':14,'qtype_changed_count':12,
 'note':'Six weight and six bias DFP descriptors become equivalent affine descriptors; PC axes follow reshaped Conv weights. Quantization files are not byte-identical.'},
 'input_fixture_path':str(input_path),'input_fixture_sha256':H(input_path),'input_bytes':input_path.stat().st_size,
 'reference_fixture_path':str(fixture_path),'reference_fixture_sha256':H(fixture_path),
 'probe_history_tokens':128,'probe_query_rows':256,'history_source':F['history_source'],
 'timing_scope':'Single fused two-layer block. Not whole 24-layer prefill or TTFT. Query rows are not user dynamic-body tokens; chat adds eight template tokens.',
 'binary_port_query_method':'SDK vxImportKernelFromURL then vxQueryKernel/vxQueryParameter/vxQueryMetaFormatAttribute; no execution graph run.',
 'binary_port_query_source_sha256':H(R/'inspect_nbg.c'),'binary_port_query_raw_sha256':H(R/'bitcast-binary-ports.json'),
 'original_binary_port_query_raw_sha256':H(R/'plain-binary-ports.json'),'board_test_status':'not_run','source_sha256':sources,
 'notes':['M32 BITCAST was bitwise equal on board but about 25% slower; M256 is a separate bounded batch-efficiency experiment.',
 'Host generation time is not NPU board inference time.', 'DFP is interpreted by fl; SCALE/ZERO_POINT union aliases are not active quantization fields.',
 'All five board outputs, A-B-A timing and driver release must be checked before acceptance.']}
(R/'binary-ports-normalized.json').write_text(json.dumps(norm,indent=2)+'\n')
(R/'deployment-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'validation':'passed','manifest_path':str(R/'deployment-manifest.json'),'manifest_sha256':H(R/'deployment-manifest.json'),'nb_sha256':m['nb_sha256'],'nb_bytes':m['nb_bytes']}))
