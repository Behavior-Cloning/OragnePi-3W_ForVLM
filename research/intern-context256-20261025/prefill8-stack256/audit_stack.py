#!/usr/bin/env python3
from pathlib import Path
import copy,hashlib,json,sys,numpy as np,yaml
R=Path(__file__).resolve().parent;REPO=R.parent.parents[1]
sys.path.insert(0,str(REPO/'npu/kv-cache'));from fused_contract import verify_contract
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def apply(g,changes):
 g=copy.deepcopy(g)
 for c in changes:
  path=c['path'].strip('/').split('/');node=g
  for part in path[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
  k=int(path[-1]) if isinstance(node,list) else path[-1];assert node[k]==c['before'];node[k]=c['after']
 return g
state=json.loads((R/'status.json').read_text());assert state['state']=='complete' and len(state['blocks'])==12
plan=json.loads((R/'plan.json').read_text());rows=[]
for item in state['blocks']:
 src=Path(item['source_ctx256_t16']);old=Path(item['reference_ctx1280_t8']);out=R/'cases'/('l'+str(item['first']))
 g=json.loads((out/'decoder.json').read_text())
 assert g==apply(json.loads((src/'decoder.json').read_text()),plan['M_changes_from_c256_t16'])
 assert g==apply(json.loads((old/'decoder.json').read_text()),plan['capacity_changes_from_c1280_t8'])
 for name,h in item['source_hashes'].items():assert sha(src/name)==h
 for name in ['decoder.quantize','decoder.data','export/decoder.export.data']:assert sha(out/name)==sha(src/name)
 assert (out/'decoder.data').resolve()==(src/'decoder.data').resolve()
 assert (out/'export/decoder.export.data').resolve()==(src/'export/decoder.export.data').resolve()
 arrays=json.loads((out/'same-source-audit.json').read_text())['arrays'];weights=np.load(out/'decoder.data',allow_pickle=True).item()
 for row in arrays:
  v=np.asarray(weights[row['node']][row['field']]);assert list(v.shape)==row['shape'] and str(v.dtype)==row['dtype'] and v.size==row['elements'] and hashlib.sha256(v.tobytes()).hexdigest()==row['sha256']
 assert len(arrays)==sum(len(x) for x in weights.values());del weights
 meta=yaml.safe_load((out/'tensor-inputmeta.yml').read_text())
 for port in meta['input_meta']['databases'][0]['ports']:assert port['shape']==g['Layers'][port['lid']]['parameters']['shape']
 c=json.loads((out/'conversion.json').read_text());assert c['exit_code']==0 and c['capacity']==256 and c['query_tokens']==8 and c['first_layer']==item['first']
 assert verify_contract(out,2)['contract_verified_from_exported_c']
 nb=out/'export_nbg_unify/network_binary.nb';assert sha(nb)==item['binary_sha256']
 rows.append({'first_layer':item['first'],'nb':str(nb),'nb_sha256':sha(nb),'bytes':nb.stat().st_size,'M_only_from_c256_t16':True,'capacity_only_from_c1280_t8':True,'quantizer_byte_equal':True,'all_weight_arrays_verified':len(arrays),'weight_elements':sum(x['elements'] for x in arrays),'raw_data_sha256':sha(out/'decoder.data'),'actual_quantized_payload_sha256':sha(out/'export/decoder.export.data'),'actual_quantized_payload_byte_equal':True,'input_output_contract':'11inputs/5outputs verified from exported C'})
report={'valid':True,'blocks':12,'capacity':256,'query_tokens':8,'fusion_width':2,'source_weights_and_quantization_changed':False,'numeric_and_performance_status':'requires parent NPU full-model comparison; no board operation from compiler subtask','rows':rows,'all_binary_bytes':sum(x['bytes'] for x in rows),'source_plan_sha256':sha(R/'plan.json'),'elapsed_build_seconds':state['ended_unix']-state['started_unix'],'space_end':state['space_end']}
(R/'full-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
