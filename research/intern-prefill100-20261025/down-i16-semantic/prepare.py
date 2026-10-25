from pathlib import Path
import numpy as np,torch,gguf,json,hashlib,sys,time,traceback
D=Path(__file__).resolve().parent;T=D.parent;R=T.parents[1];sys.path.insert(0,str(R/'npu/kv-cache'))
from trained_probe import TrainedDecoder
from exported_int8 import apply_exported_int8
S=T/'graphs/cases/c512-m32-l2';F=T/'fullconv-quality/semantic-inputs.json';M=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state={'status':'preparing','started':time.time(),'cases':[],'scope':'48 windows from24 prompts/12 scenarios, source FP32 history; held out from single-trajectory down calibration; not full-model task quality'}
def save():(D/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 torch.set_num_threads(2);assert sha(M)=='e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f'
 reader=gguf.GGUFReader(M);embedding=next(t for t in reader.tensors if t.name=='token_embd.weight');emb=gguf.dequantize(embedding.data,embedding.tensor_type)
 layers=[TrainedDecoder(reader,i,True).eval() for i in range(4)];qlayers=[TrainedDecoder(reader,i,True).eval() for i in [2,3]]
 for i,l in enumerate(qlayers):apply_exported_int8(l,S,prefix=f'b{i}_')
 conv=json.loads((S/'conversion.json').read_text());g=json.loads((S/'decoder.json').read_text());qs=json.loads(F.read_text())['quality'];assert len(qs)==24
 with torch.inference_mode():
  for q in qs:
   ids=q['tokens'];assert len(ids)==q['prompt_tokens'] and len(ids)==q['static_tokens']+q['body_tokens']+8
   assert hashlib.sha256(np.array(ids,dtype='<u4').tobytes()).hexdigest()==q['tokens_sha256']
   h=torch.from_numpy(emb[ids][None].copy())
   for l in layers[:2]:h=l.full(h)
   hidden=h;kv=[]
   for l in layers[2:]:
    _,k,v=l.project(h,*l.rope(torch.arange(len(ids))));kv.append((k,v));h=l.full(h)
   for name,pos in [('first',q['static_tokens']),('last',len(ids)-32)]:
    assert pos+32<=len(ids) and pos<=512
    out=D/(q['id']+'-'+name);out.mkdir();feeds={'b0_hidden':hidden[:,pos:pos+32].clone()}
    for i,l in enumerate(layers[2:]):
     pk=torch.zeros(1,l.kv,512,l.hd);pv=pk.clone();pk[:,:,:pos]=kv[i][0][:,:,:pos];pv[:,:,:pos]=kv[i][1][:,:,:pos]
     mask=torch.full((1,1,32,544),-10000.);mask[:,:,:,:pos]=0
     for j in range(32):mask[:,:,j,512:512+j+1]=0
     mask=mask.repeat(1,1,l.heads//l.kv,1);cos,sin=l.rope(torch.arange(pos,pos+32))
     feeds.update({f'b{i}_'+k:v for k,v in dict(past_k=pk,past_v=pv,mask=mask,rope_cos=cos,rope_sin=sin).items()})
    expected={};x=feeds['b0_hidden']
    for i,l in enumerate(qlayers):
     f={k:feeds[f'b{i}_'+k] for k in ['past_k','past_v','rope_cos','rope_sin']};x,k,v=l(x,f['past_k'],f['past_v'],feeds[f'b{i}_mask'][:,:,:32],f['rope_cos'],f['rope_sin']);expected.update({f'b{i}_attach_new_k/out0':k,f'b{i}_attach_new_v/out0':v})
    expected['b1_attach_hidden_out/out0']=x
    inorder=[g['Layers'][k]['name'] for k in conv['input_order']];outorder=[g['Layers'][k]['name'] for k in conv['output_order']]
    for k,v in feeds.items():assert torch.isfinite(v).all(),k
    np.concatenate([feeds[k].numpy().ravel() for k in inorder]).astype('<f4').tofile(out/'input.f32')
    np.concatenate([expected[k].numpy().ravel() for k in outorder]).astype('<f4').tofile(out/'reference.f32')
    record={'id':out.name,'scenario':q['case'],'window':name,'history':pos,'query_tokens':32,'capacity':512,'first_layer':2,'fusion_width':2,'static_tokens':q['static_tokens'],'system_body_tokens':q['system_body_tokens'],'system_template_tokens':5,'body_tokens':q['body_tokens'],'dynamic_template_tokens':8,'prompt_tokens':len(ids),'input_sha256':sha(out/'input.f32'),'reference_sha256':sha(out/'reference.f32'),'tokens_sha256':q['tokens_sha256'],'source':'FP32 Q8-derived full-depth prefix, not captured NPU history; original exported-weight FP32 reference','claim':'single-block numerical screening only'}
    (out/'fixture.json').write_text(json.dumps(record,indent=2)+'\n');state['cases'].append(record)
   save();print(q['id'],flush=True)
 state.update(status='complete',ended=time.time(),fixture_source_sha256=sha(F),model_sha256=sha(M),generator_sha256=sha(Path(__file__)));save()
except BaseException:state.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
