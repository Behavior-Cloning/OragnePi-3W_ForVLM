import struct,pathlib,json,hashlib,collections,math,shutil
P=pathlib.Path(__file__).parent
S={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def inspect(path):
 with path.open('rb') as f:
  def unpack(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
  def string():return f.read(unpack('Q')).decode('utf-8')
  def val(t,keep=True):
   if t==8:
    size=unpack('Q')
    if keep:return f.read(size).decode('utf-8')
    f.seek(size,1);return None
   if t==9:
    child=unpack('I');n=unpack('Q')
    if child in S:
     f.seek(n*struct.calcsize('<'+S[child]),1);return dict(element_type=child,count=n)
    for _ in range(n):val(child,False)
    return dict(element_type=child,count=n)
   if t in S:return unpack(S[t])
   raise ValueError('unsupported GGUF metadata type '+str(t))
  assert f.read(4)==b'GGUF'
  version=unpack('I');nt=unpack('Q');nk=unpack('Q');meta={};tok_hash={}
  for _ in range(nk):
   key=string();typ=unpack('I');start=f.tell();value=val(typ,not key.startswith('tokenizer.'));end=f.tell()
   if key.startswith('tokenizer.'):
    f.seek(start);tok_hash[key]=hashlib.sha256(f.read(end-start)).hexdigest();f.seek(end)
   elif key.startswith(('llama.','general.')):meta[key]=value
  tensors=[];counts=collections.Counter()
  for _ in range(nt):
   name=string();nd=unpack('I');shape=[unpack('Q') for _ in range(nd)];typ=unpack('I');offset=unpack('Q')
   tensors.append(dict(name=name,shape=shape,type_code=typ,offset=offset));counts[str(typ)]+=1
  return dict(file=str(path),bytes=path.stat().st_size,gguf_version=version,tensor_count=nt,metadata=meta,tokenizer_field_sha256=tok_hash,tensor_type_counts_by_code=dict(counts),stored_tensor_elements=sum(math.prod(t['shape']) for t in tensors),tensors=tensors)
q8=P/'FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf';q4=P.parent/'FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf'
a=inspect(q8);b=inspect(q4)
assert a['metadata']['general.architecture']=='llama'
assert a['metadata']['llama.block_count']==16
keys=[k for k in a['metadata'] if k.startswith('llama.')]
checks=dict(same_architecture=a['metadata']['general.architecture']==b['metadata']['general.architecture'],same_llama_config=all(a['metadata'][k]==b['metadata'].get(k) for k in keys),same_tokenizer_metadata=a['tokenizer_field_sha256']==b['tokenizer_field_sha256'],same_tensor_names_and_shapes=[(t['name'],t['shape']) for t in a['tensors']]==[(t['name'],t['shape']) for t in b['tensors']],q8_tensor_type_present=a['tensor_type_counts_by_code'].get('8',0)>0)
assert all(checks.values()),checks
(P/'gguf-inspection.json').write_text(json.dumps(a,indent=2));(P/'q4-q8-structural-parity.json').write_text(json.dumps(dict(checks=checks,Q4_source=str(q4),Q8_source=str(q8),Q4_type_counts=b['tensor_type_counts_by_code'],Q8_type_counts=a['tensor_type_counts_by_code'],config={k:a['metadata'][k] for k in keys},F_free_bytes=shutil.disk_usage(P).free),indent=2))
print(json.dumps(dict(checks=checks,metadata=a['metadata'],tensor_type_counts=a['tensor_type_counts_by_code'],stored_tensor_elements=a['stored_tensor_elements'],F_free_bytes=shutil.disk_usage(P).free),indent=2))
