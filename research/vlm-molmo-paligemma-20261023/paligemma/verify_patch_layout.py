import pathlib,json,torch,safetensors
from PIL import Image
from transformers import AutoProcessor
P=pathlib.Path(__file__).parent;torch.set_num_threads(2)
processor=AutoProcessor.from_pretrained(P/'model',local_files_only=True)
x=processor.image_processor(images=Image.open(P/'gemma.cpp-main/paligemma/testdata/image.ppm'),return_tensors='pt')['pixel_values'].float()
idx=json.loads((P/'model/model.safetensors.index.json').read_text())['weight_map'];wk='vision_tower.vision_model.embeddings.patch_embedding.weight';bk='vision_tower.vision_model.embeddings.patch_embedding.bias'
with safetensors.safe_open(P/'model'/idx[wk],framework='pt') as f:w=f.get_tensor(wk).float()
with safetensors.safe_open(P/'model'/idx[bk],framework='pt') as f:b=f.get_tensor(bk).float()
y=torch.nn.functional.conv2d(x,w,b,stride=14).permute(0,2,3,1).reshape(-1,1152)
patch=x.unfold(2,14,14).unfold(3,14,14).permute(0,2,3,4,5,1).contiguous().reshape(-1,588)
ohwi=w.permute(0,2,3,1).contiguous().reshape(1152,588);z=patch@ohwi.T+b
r=dict(input_shape=list(x.shape),kernel_shape=list(w.shape),patch_matrix=list(patch.shape),output_shape=list(y.shape),max_abs=float((y-z).abs().max()),relative_l2=float(torch.norm(y-z)/torch.norm(y)),verified_layout='OIHW -> OHWI -> [out=1152, patchH*patchW*channels=588]');assert torch.allclose(y,z,atol=2e-4,rtol=2e-4);r['passed']=True;(P/'patch-embedding-layout-parity.json').write_text(json.dumps(r,indent=2));print(r)
