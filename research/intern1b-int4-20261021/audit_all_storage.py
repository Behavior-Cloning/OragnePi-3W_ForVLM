from pathlib import Path
import json,re,numpy as np
r=Path(__file__).resolve().parent;root=Path('/opt/opi-lab/intern1b-int4-20261021');records=json.loads((r/'compile-progress.json').read_text())['records'];rows=[]
for rec in records:
 p=root/rec['kind']/('l'+str(rec['layer']));c=(p/'export/vnn_decoder.c').read_text();matches=re.findall(r'NEW_CONST_TENSOR\(const_tensor\[\d+\], attr, VSI_NN_TYPE_INT8, (\d+), (\d+)\)',c);values=[]
 with (p/'export/decoder.export.data').open('rb') as f:
  for offset,size in matches:
   f.seek(int(offset));a=np.frombuffer(f.read(int(size)),np.int8);values.append({'min':int(a.min()),'max':int(a.max()),'bytes':len(a)})
 rows.append({'kind':rec['kind'],'layer':rec['layer'],'requested_w4_matrices':len(rec['changed']),'exported_int8_constants':len(values),'all_int8_within_4bit_range':all(x['min']>=-7 and x['max']<=7 for x in values),'int4_c_identifiers':c.count('VSI_NN_TYPE_INT4'),'constants':values})
assert all(x['requested_w4_matrices']==x['exported_int8_constants'] and x['all_int8_within_4bit_range'] for x in rows)
out={'graphs':len(rows),'w4_matrices_including_prefill_duplicates':sum(x['requested_w4_matrices'] for x in rows),'all_verified':True,'conclusion':'All requested W4 FC constants export as INT8 values -7..7, not packed INT4 C tensors. NBG internal packing/arithmetic remains unproven.','rows':rows};(r/'all-graph-storage-audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='rows'})
