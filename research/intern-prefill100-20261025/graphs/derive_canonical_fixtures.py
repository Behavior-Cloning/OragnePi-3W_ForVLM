#!/usr/bin/env python3
"""One frozen FP32 history per fused block, exact shared feeds across all M.
Copies no model weights; canonical fixtures are independent immutable outputs.
"""
from pathlib import Path
import hashlib,json,sys,time,traceback
import gguf,numpy as np,torch
R=Path(__file__).resolve().parent;TASK=R.parent;REPO=TASK.parents[1];OUT=TASK/'fixtures-canonical'
sys.path.insert(0,str(REPO/'npu/kv-cache'))
from trained_probe import TrainedDecoder
from exported_int8 import apply_exported_int8
from graph_reference import evaluate
MODEL=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def jwrite(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def rel(a,b):return float(torch.linalg.vector_norm(a-b)/torch.clamp(torch.linalg.vector_norm(b),min=1e-30))
state={'status':'running','started_unix':time.time(),'cases':[]}
def save():jwrite(R/'canonical-fixture-status.json',state)
save()
try:
 assert not OUT.exists(),'canonical output already exists; preserve it'
 OUT.mkdir()
 assert sha(MODEL)=='e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f'
 torch.set_num_threads(2);reader=gguf.GGUFReader(MODEL)
 manifest=json.loads((R/'deployment-manifest.json').read_text());cases={c['id']:c for c in manifest['cases']}
 report={'scope':'single-block numerical/runtime fixtures; not full-model prefill or TTFT; history is independent source-Q8 FP32, not captured NPU KV','canonical_source':'one original M256 fixture per first_layer; all smaller M slice exact bytes from it','history':128,'capacity':512,'first_layers':[0,2],'source_model_sha256':sha(MODEL),'generator_sha256':sha(Path(__file__)),'helper_source_sha256':{n:sha(REPO/'npu/kv-cache'/n) for n in ['trained_probe.py','exported_int8.py','graph_reference.py']},'cases':[]}
 with torch.inference_mode():
  for first in [0,2]:
   original=TASK/'fixtures'/f'c512-m256-l{first}';meta=json.loads((original/'probe-fixture.json').read_text());assert meta['history']==128 and meta['query_tokens']==256 and meta['capacity']==512
   raw=np.fromfile(original/'probe-input.f32',np.float32);basefolder=R/'cases'/f'c512-m256-l{first}';basegraph=json.loads((basefolder/'decoder.json').read_text());baseconv=json.loads((basefolder/'conversion.json').read_text())
   assert baseconv['first_layer']==first and baseconv['fusion_width']==2
   full={};offset=0
   for key,num in meta['inputs']:
    node=next(v for v in basegraph['Layers'].values() if v.get('op')=='input' and v['name']==key);shape=node['parameters']['shape'];assert int(np.prod(shape))==num
    full[key]=torch.from_numpy(raw[offset:offset+num].copy().reshape(shape));offset+=num
   assert offset==raw.size and np.isfinite(raw).all()
   source_input_sha=sha(original/'probe-input.f32');base_fp=None;base_quant=None
   for m in [256,16,32,64,128]:
    name=f'c512-m{m}-l{first}';folder=R/'cases'/name;out=OUT/name;out.mkdir()
    conv=json.loads((folder/'conversion.json').read_text());graph=json.loads((folder/'decoder.json').read_text());data=np.load(folder/'decoder.data',allow_pickle=True).item()
    assert conv['first_layer']==first and conv['capacity']==512 and conv['query_tokens']==m
    feeds={};input_same=[]
    for key,v in full.items():
     if key=='b0_hidden':x=v[:,:m].clone()
     elif '_past_' in key:x=v.clone()
     elif '_rope_' in key:x=v[:,:,:m].clone()
     elif key.endswith('_mask'):
      rows=v.reshape(1,1,7,256,768)[:,:,:,:m]
      x=torch.cat([rows[:,:,:,:,:512],rows[:,:,:,:,512:512+m]],dim=-1).reshape(1,1,7*m,512+m).clone()
      expected=torch.full((1,1,m,512+m),-10000.);expected[:,:,:,:128]=0
      for q in range(m):expected[:,:,q,512:512+q+1]=0
      expected=expected.repeat(1,1,7,1);assert torch.equal(x,expected),(name,key,'causal mask')
     else:raise ValueError(key)
     expectedshape=next(v['parameters']['shape'] for v in graph['Layers'].values() if v.get('op')=='input' and v['name']==key)
     assert list(x.shape)==expectedshape and torch.isfinite(x).all(),(name,key,list(x.shape),expectedshape)
     feeds[key]=x;input_same.append({'name':key,'shape':list(x.shape),'f32_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'canonical_same_values':True,'operation':'unchanged full past' if '_past_' in key else 'exact query/mask slice'})
    fp=evaluate(graph,data,feeds);del data
    x=feeds['b0_hidden'];quant={};records=[]
    for i in range(2):
     layer=TrainedDecoder(reader,first+i,True).eval();records.append(apply_exported_int8(layer,folder,prefix=f'b{i}_'))
     f={k:feeds[f'b{i}_'+k] for k in ['past_k','past_v','rope_cos','rope_sin']};mask=feeds[f'b{i}_mask'][:,:,:m]
     x,k,v=layer(x,f['past_k'],f['past_v'],mask,f['rope_cos'],f['rope_sin']);quant.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
    quant['b1_attach_hidden_out/out0']=x
    if m==256:base_fp={k:v.clone() for k,v in fp.items()};base_quant={k:v.clone() for k,v in quant.items()}
    errors={};quant_errors={}
    for key,value in fp.items():
     ref=base_fp[key][:,:m] if 'hidden' in key else base_fp[key][:,:,:m]
     err=rel(value,ref);torch.testing.assert_close(value,ref,atol=1e-4,rtol=5e-4);assert err<1e-5,(name,key,err)
     errors[key]={'relative_l2':err,'max_abs':float((value-ref).abs().max()),'finite':bool(torch.isfinite(value).all())}
     qref=base_quant[key][:,:m] if 'hidden' in key else base_quant[key][:,:,:m]
     qerr=rel(quant[key],qref);torch.testing.assert_close(quant[key],qref,atol=1e-4,rtol=5e-4);assert qerr<1e-5
     quant_errors[key]={'relative_l2':qerr,'max_abs':float((quant[key]-qref).abs().max()),'finite':bool(torch.isfinite(quant[key]).all())}
    inorder=[graph['Layers'][k]['name'] for k in conv['input_order']];outorder=[graph['Layers'][k]['name'] for k in conv['output_order']]
    assert len(inorder)==11 and len(outorder)==5
    np.concatenate([feeds[k].numpy().ravel() for k in inorder]).tofile(out/'probe-input.f32')
    np.concatenate([quant[k].numpy().ravel() for k in outorder]).tofile(out/'probe-reference.f32')
    np.concatenate([fp[k].numpy().ravel() for k in outorder]).tofile(out/'probe-source-fp32.f32')
    entry={'id':name,'first_layer':first,'fusion_width':2,'history':128,'capacity':512,'query_tokens':m,'token_ids':meta['token_ids'][:128+m],'canonical_source':str(original),'canonical_source_input_sha256':source_input_sha,'canonical_source_tokens_sha256':hashlib.sha256(np.array(meta['token_ids'],dtype='<u4').tobytes()).hexdigest(),'history_source':'independent FP32 full-depth prefix; not captured NPU KV','nb_sha256':sha(folder/'export_nbg_unify/network_binary.nb'),'input_sha256':sha(out/'probe-input.f32'),'reference_sha256':sha(out/'probe-reference.f32'),'source_fp32_sha256':sha(out/'probe-source-fp32.f32'),'inputs':[(k,feeds[k].numel()) for k in inorder],'outputs':[(k,quant[k].numel()) for k in outorder],'input_contract':input_same,'fp32_graph_same_effective_rows':errors,'exported_weight_fp32_same_effective_rows':quant_errors,'exported_weight_records':records,'timing_scope':'canonical fixture generation and references excluded from board measurements'}
    for filename in ['decoder.json','decoder.data','decoder.quantize','conversion.json','export','export_nbg_unify']:
     target=(folder/filename).resolve();(out/filename).symlink_to(target,target_is_directory=target.is_dir())
    jwrite(out/'probe-fixture.json',entry);report['cases'].append(entry);state['cases'].append({'id':name,'input_sha256':entry['input_sha256'],'reference_sha256':entry['reference_sha256'],'max_fp32_relL2':max(v['relative_l2'] for v in errors.values()),'max_quant_weight_relL2':max(v['relative_l2'] for v in quant_errors.values())});save();print(name,'canonical PASS',flush=True)
 jwrite(R/'canonical-fixture-manifest.json',report);state.update(status='complete',ended_unix=time.time(),manifest_sha256=sha(R/'canonical-fixture-manifest.json'));save()
except BaseException:
 state.update(status='failed',ended_unix=time.time(),error=traceback.format_exc());save();raise
