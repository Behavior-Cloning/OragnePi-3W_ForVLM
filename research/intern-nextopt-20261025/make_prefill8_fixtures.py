"""Frozen Chinese tail fixtures: genuine source-weight FP32 history, not captured NPU KV."""
import hashlib,json,sys
from pathlib import Path
import gguf,numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'fixture-source'))
from trained_probe import TrainedDecoder
from exported_int8 import apply_exported_int8
from graph_reference import evaluate
REPO=ROOT.parents[1];BUILD=Path('/opt/opi-lab/intern-nextopt-20261025')
MODEL=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
FOLDERS={16:Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l0'),8:BUILD/'prefill8/l0'}
OUT=BUILD/'fixtures';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(MODEL)=='e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f'
torch.set_num_threads(2);reader=gguf.GGUFReader(MODEL)
t=next(t for t in reader.tensors if t.name=='token_embd.weight');emb=gguf.dequantize(t.data,t.tensor_type)
fp=[TrainedDecoder(reader,i,True).eval() for i in range(2)]
quant={n:[TrainedDecoder(reader,i,True).eval() for i in range(2)] for n in FOLDERS}
weight_records={}
for n,folder in FOLDERS.items():
 weight_records[n]=[apply_exported_int8(layer,folder,prefix=f'b{i}_') for i,layer in enumerate(quant[n])]
for i in range(2):
 for (name,x),(name2,y) in zip(quant[8][i].named_buffers(),quant[16][i].named_buffers()):
  assert name==name2 and torch.equal(x,y),(i,name)
inputs_path=REPO/'research/async-agent-20261025/aligned-inputs.json'
inputs=json.loads(inputs_path.read_text());cases=[x for x in inputs['performance'] if x['id'].endswith('-0')]
assert len(cases)==4
report={'source_model':str(MODEL),'model_sha256':sha(MODEL),'input_manifest_sha256':sha(inputs_path),
 'history_scope':'Real nonzero source-Q8-dequantized FP32 causal history at each layer; not captured deployed NPU KV. No image/audio/full-language-model quality tested.',
 'pair_scope':'Same first8 valid rows, same historical KV, mask semantics and RoPE positions; t16 last8 use last-token padding as resident runtime.',
 'capacity':1280,'first_layer':0,'fusion_width':2,'weight_records':weight_records,'cases':[]}
with torch.inference_mode():
 for case in cases:
  tokens=case['tokens'];pos=len(tokens)-8;assert pos in (96,128,160,192)
  hidden=torch.from_numpy(emb[tokens][None].copy());past=[];full=hidden
  for layer in fp:
   _,k,v=layer.project(full,*layer.rope(torch.arange(len(tokens))))
   pk=torch.zeros(1,2,1280,64);pv=pk.clone();pk[:,:,:pos]=k[:,:,:pos];pv[:,:,:pos]=v[:,:,:pos]
   assert torch.count_nonzero(pk)>1000 and torch.count_nonzero(pv)>1000
   past.append((pk,pv));full=layer.full(full)
  entry={'id':case['id'],'static_tokens':case['static_tokens'],'system_body_tokens':case['system_body_tokens'],
    'system_template_tokens':case['system_template_tokens'],'body_tokens':case['body_tokens'],'dynamic_template_tokens':case['template_overhead'],
    'prompt_tokens':len(tokens),'history':pos,'valid_tail_tokens':8,'token_ids':tokens,'variants':{}}
  refs={}
  for n,folder in FOLDERS.items():
   loc=OUT/case['id']/f't{n}';loc.mkdir(parents=True,exist_ok=True)
   conv=json.loads((folder/'conversion.json').read_text());graph=json.loads((folder/'decoder.json').read_text())
   data=np.load(folder/'decoder.data',allow_pickle=True).item();x=hidden[:,pos:].clone()
   if n==16:x=torch.cat([x,x[:,-1:].repeat(1,8,1)],1)
   feeds={'b0_hidden':x};expected={};qx=x
   for i in range(2):
    mask=torch.full((1,1,n,1280+n),-10000.);mask[:,:,:,:pos]=0
    for j in range(n):mask[:,:,j,1280:1280+j+1]=0
    cos,sin=fp[i].rope(torch.arange(pos,pos+n));pk,pv=past[i]
    feeds.update({f'b{i}_past_k':pk,f'b{i}_past_v':pv,f'b{i}_mask':mask.repeat(1,1,7,1),f'b{i}_rope_cos':cos,f'b{i}_rope_sin':sin})
    qx,k,v=quant[n][i](qx,pk,pv,mask,cos,sin)
    expected.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
   expected['b1_attach_hidden_out/out0']=qx
   graph_outputs=evaluate(graph,data,feeds)
   fp_x=x;fp_expected={}
   for i in range(2):
    fp_x,k,v=fp[i](fp_x,past[i][0],past[i][1],feeds[f'b{i}_mask'][:,:,:n],feeds[f'b{i}_rope_cos'],feeds[f'b{i}_rope_sin'])
    fp_expected.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
   fp_expected['b1_attach_hidden_out/out0']=fp_x
   errs={}
   for name,value in graph_outputs.items():
    y=fp_expected[name];err=float(torch.linalg.vector_norm(value-y)/torch.linalg.vector_norm(y));assert err<1e-5,(name,err);errs[name]=err
   inp=[graph['Layers'][k]['name'] for k in conv['input_order']];out=[graph['Layers'][k]['name'] for k in conv['output_order']]
   np.concatenate([feeds[k].numpy().ravel() for k in inp]).tofile(loc/'input.f32')
   np.concatenate([expected[k].numpy().ravel() for k in out]).tofile(loc/'reference.f32')
   refs[n]={k:v.numpy() for k,v in expected.items()}
   meta={'query_tokens':n,'input_order':[{'name':k,'shape':list(feeds[k].shape),'count':feeds[k].numel()} for k in inp],
    'output_order':[{'name':k,'shape':list(expected[k].shape),'count':expected[k].numel()} for k in out],
    'input_sha256':sha(loc/'input.f32'),'reference_sha256':sha(loc/'reference.f32'),'nbg_sha256':sha(folder/'export_nbg_unify/network_binary.nb'),
    'host_fp32_graph_relative_l2':errs,'past_k_nonzero':[int(torch.count_nonzero(z[0])) for z in past],'past_v_nonzero':[int(torch.count_nonzero(z[1])) for z in past]}
   (loc/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n');entry['variants'][str(n)]=meta
  parity={}
  for k in refs[8]:
   a=refs[8][k];b=refs[16][k][:,:8] if 'hidden' in k else refs[16][k][:,:,:8]
   err=float(np.linalg.norm(a-b)/np.linalg.norm(b));assert err<1e-5,(case['id'],k,err);parity[k]=err
  entry['host_exported_weight_first8_relative_l2']=parity;report['cases'].append(entry)
  print(case['id'],pos,'ok',flush=True)
(ROOT/'prefill8-fixtures.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('ALL_FIXTURES_COMPLETE',flush=True)
