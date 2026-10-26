from pathlib import Path
import sys,json,hashlib,time,numpy as np
sys.path.insert(0,'/opt/opi-lab/builds/intern-frontmerge-20261027-r1/source/gguf-py')
import gguf
D=Path(__file__).resolve().parent;M=Path('/mnt/f/Files/OPi-Lab/models/internvl3-1b-vision')
src=M/'mmproj-InternVL3-1B-Instruct-f16.gguf';out=M/'mmproj-InternVL3-1B-Instruct-encoder-Q8_0-projector-F16-r1.gguf'
assert src.exists() and not out.exists()
r=gguf.GGUFReader(src);arch=r.fields['general.architecture'].contents();w=gguf.GGUFWriter(out,arch=arch,use_temp_file=True)
for f in r.fields.values():
 if f.name in ['general.architecture','general.file_type','general.quantization_version'] or f.name.startswith('GGUF.'):continue
 typ=f.types[0];sub=f.types[-1] if typ==gguf.GGUFValueType.ARRAY else None
 w.add_key_value(f.name,f.contents(),typ,sub_type=sub)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0.value)
w.add_quantization_version(gguf.GGML_QUANT_VERSION)
audit=[]
for t in r.tensors:
 change=t.name.startswith('v.blk.') and t.name.endswith('.weight') and len(t.shape)==2
 if change:
  assert t.tensor_type in (gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.F32)
  a=t.data.astype(np.float32);q=gguf.quants.quantize(a,gguf.GGMLQuantizationType.Q8_0);rt=gguf.quants.dequantize(q,gguf.GGMLQuantizationType.Q8_0)
  w.add_tensor(t.name,q,raw_dtype=gguf.GGMLQuantizationType.Q8_0)
  audit.append({'name':t.name,'shape':list(map(int,t.shape)),'from':str(t.tensor_type),'to':'Q8_0','source_bytes':t.n_bytes,'bytes':q.nbytes,'relative_l2':float(np.linalg.norm(rt-a)/max(np.linalg.norm(a),1e-12))})
 else:w.add_tensor(t.name,t.data,raw_dtype=t.tensor_type)
assert len(audit)==144,len(audit)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
r2=gguf.GGUFReader(out);assert len(r2.tensors)==len(r.tensors)
changed={x['name'] for x in audit};unchanged=0
for a,b in zip(r.tensors,r2.tensors):
 assert a.name==b.name and np.array_equal(a.shape,b.shape)
 if a.name in changed:assert b.tensor_type==gguf.GGMLQuantizationType.Q8_0
 else:assert a.tensor_type==b.tensor_type and np.array_equal(a.data,b.data);unchanged+=1
for k,f in r.fields.items():
 if k.startswith('GGUF.') or k in ['general.file_type','general.quantization_version']:continue
 assert f.contents()==r2.fields[k].contents(),k
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
s={'state':'quantized_unvalidated','scope':'144 ViT block linear matrices only; patch stem/projector/position/norm/bias unchanged byte-for-byte; non-quantization metadata preserved; file_type and quantization_version updated; tensor types are authoritative','source':str(src),'source_sha256':sha(src),'output':str(out),'output_sha256':sha(out),'bytes':out.stat().st_size,'unchanged_tensors':unchanged,'changed_tensors':len(audit),'weights':audit,'time':time.time()}
(D/'vision-q8-manifest.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:v for k,v in s.items() if k!='weights'}))
