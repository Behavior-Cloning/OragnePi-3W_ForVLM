from pathlib import Path
import numpy as np,yaml,re,json
root=Path('/opt/opi-lab/intern1b-int4-20261021/decode/l0');q=yaml.safe_load((root/'decoder.quantize').read_text())['quantize_parameters'];d=np.load(root/'decoder.data',allow_pickle=True).item();c=(root/'export/vnn_decoder.c').read_text();blob=(root/'export/decoder.export.data').read_bytes();rows=[]
for m in re.finditer(r'/\* @([^:]+):weight \*/(.*?)(?=/\*|$)',c,re.S):
 name,block=m.groups()
 if 'fullconnect' not in name:continue
 hit=re.search(r'NEW_CONST_TENSOR\(const_tensor\[\d+\], attr, VSI_NN_TYPE_INT8, (\d+), (\d+)\)',block)
 if not hit:continue
 offset,size=map(int,hit.groups());scalename=re.search(r'attr.dtype.scales = (scales_\d+);',block).group(1);scaletext=re.search(r'float\s+'+scalename+r'\[\] = \{(.*?)\};',c,re.S).group(1);scales=np.array([float(x) for x in scaletext.split(',')],np.float32)
 orig=d[name]['weight'];actual=np.frombuffer(blob[offset:offset+size],np.int8).reshape(orig.T.shape).T;recon=actual*scales;expected=np.clip(np.rint(orig/np.array(q['@'+name+':weight']['scale'])), -7,7)
 rows.append({'name':name,'int_range':[int(actual.min()),int(actual.max())],'scale_ratio_min':float((scales/np.array(q['@'+name+':weight']['scale'])).min()),'scale_ratio_max':float((scales/np.array(q['@'+name+':weight']['scale'])).max()),'integer_match_expected_fraction':float(np.mean(actual==expected)),'relative_l2_actual':float(np.linalg.norm(recon-orig)/np.linalg.norm(orig))})
out={'scope':'compiler exported INT8 data interpreted using generated C column-major constant layout','rows':rows};Path(__file__).with_name('weight-reconstruction-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
