#!/usr/bin/env python3
from pathlib import Path
import copy,hashlib,json,os,sys,yaml
R=Path(__file__).resolve().parent;F=R/'full-stack256';sys.path.insert(0,str(R.parents[1]/'npu/kv-cache'))
from fused_contract import verify_contract
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
s=json.loads((F/'status.json').read_text());assert s['state']=='complete';assert len(s['cases'])==24
plan=json.loads((F/'plan.json').read_text());assert len(plan['cases'])==24
rows=[]
for item,result in zip(plan['cases'],s['cases']):
 assert item['kind']==result['kind'] and item['first_layer']==result['first_layer']
 src=Path(item['source']);out=F/'cases'/item['kind']/('l'+str(item['first_layer']))
 expected=json.loads((src/'decoder.json').read_text())
 for change in item['capacity_changes']:
  parts=change['path'].strip('/').split('/');node=expected
  for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
  key=int(parts[-1]) if isinstance(node,list) else parts[-1];assert node[key]==change['before'];node[key]=change['after']
 assert expected==json.loads((out/'decoder.json').read_text()),str(out)
 for name,h in item['source_hashes'].items():assert sha(src/name)==h,(out,name)
 assert (out/'decoder.data').is_symlink() and (out/'decoder.data').resolve()==(src/'decoder.data').resolve()
 assert sha(out/'decoder.data')==item['source_hashes']['decoder.data']
 assert sha(out/'decoder.quantize')==item['source_hashes']['decoder.quantize']
 c=json.loads((out/'conversion.json').read_text());assert c['capacity']==256 and c['query_tokens']==item['query_tokens'] and c['first_layer']==item['first_layer'] and c['exit_code']==0
 assert verify_contract(out,2)['contract_verified_from_exported_c']
 meta=yaml.safe_load((out/'tensor-inputmeta.yml').read_text())
 for port in meta['input_meta']['databases'][0]['ports']:assert port['shape']==expected['Layers'][port['lid']]['parameters']['shape']
 nb=out/'export_nbg_unify/network_binary.nb';assert sha(nb)==result['candidate_sha256']
 rows.append({'output_name':item['output_name'],'nb_sha256':sha(nb),'nb_bytes':nb.stat().st_size,'first_layer':item['first_layer'],'query_tokens':item['query_tokens'],'capacity':256,'graph_only_expected_capacity_edits':True,'quantizer_bytes_unchanged':True,'weights_exact_source_reference':True,'source_hashes':item['source_hashes'],'weight_arrays':result['weight_arrays'],'weight_elements':result['weight_elements'],'contract_verified_from_exported_c':True,'array_manifest_sha256':sha(out/'same-source-audit.json')})
files=[p for p in F.rglob('*') if p.is_file() and not p.is_symlink()];v=os.statvfs('/mnt/f')
report={'valid':True,'scope':'24 compiled replacement graphs; NPU whole-model validation still pending','graphs':24,'decode_blocks':12,'prefill16_blocks':12,'heads_embeddings_copied':False,'rows':rows,'sum_replacement_nb_bytes':sum(x['nb_bytes'] for x in rows),'full_regular_bytes':sum(p.stat().st_size for p in files),'f_available_bytes':v.f_bavail*v.f_frsize,'budget_bytes':4*1024**3,'runtime_flags_preserved':['prefill16','tail8','group','mask'],'max_new_tokens':32}
assert report['full_regular_bytes']<report['budget_bytes'];(F/'full-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
