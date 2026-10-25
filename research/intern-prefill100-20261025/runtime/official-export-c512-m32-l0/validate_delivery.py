#!/usr/bin/env python3
"""Revalidate frozen delivered files and binary metadata without board or execution."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;J=lambda f:json.loads((P/f).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=J('deployment-manifest.json');p=J('source-proof.json');audit=J('INDEPENDENT_EXPORT_AUDIT.json');meta=J('nbg_meta.expected.json')
a=J('plain-binary-ports.json');b=J('bitcast-binary-ports.json');assert a==b and a['count']==16
assert m['geometry']=={'first_layer':0,'fusion_width':2,'capacity':512,'query_tokens':32}
assert sha(m['nb_path'])==m['nb_sha256'] and Path(m['nb_path']).stat().st_size==m['nb_bytes']
assert sha(P/'plain/network_binary.nb')==m['source_nb_sha256']==p['source_nb_sha256']
for path,expected in m['sources'].items():assert sha(P/path)==expected
assert sum(x['direction']==0 for x in a['ports'])==11 and sum(x['direction']==1 for x in a['ports'])==5
for i,(port,(name,expected)) in enumerate(zip(a['ports'],list(meta['Inputs'].items())+list(meta['Outputs'].items()))):
 assert port['index']==i and port['dims']==expected['shape'][::-1]
 assert port['direction']==(0 if i<11 else 1)
 assert all(port[k]==0 for k in ['quant_status','fl_status','zp_status','scale_status'])
 if 'quantizer' in expected:assert port['dtype']==4 and port['quant_type']==1 and port['fl']==expected['quantize']['fl']
 else:assert expected['dtype']=='float16' and port['dtype']==15 and port['quant_type']==0
assert audit['audit_pass'] and audit['weights']==audit['biases']==14
w=[x for x in audit['constant_records'] if x['key'].endswith(':weight')];b=[x for x in audit['constant_records'] if x['key'].endswith(':bias')]
assert len(w)==len(b)==14 and all(x['raw_and_numeric_equal'] for x in w+b) and all(x['dtype']=='INT8' for x in w)
assert audit['all_constant_integer_bytes_and_numeric_values_equal']
assert all(m[k] is True for k in ['all_external_ports_exact','all_14_weight_values_exact','all_14_weight_integer_codes_exact','all_14_bias_values_exact','all_external_quant_grids_equal'])
print(json.dumps({'status':'all_frozen_delivery_assertions_pass','manifest_sha256':sha(P/'deployment-manifest.json'),'actual_input_ports':11,'actual_output_ports':5,'weights':14,'biases':14}))
