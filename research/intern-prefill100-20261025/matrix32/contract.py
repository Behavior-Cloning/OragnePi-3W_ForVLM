"""Original precision C512 M16/M32/M16-repeat deployment contract."""
from pathlib import Path
import hashlib,json,os,re
BASE='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018'
BINARY_SHA='bf19922fb62bdde70a15ccde742abf9b4ffa6e5b116e73929fa7310442dc1e11'
NAMES=['original-m16','original-m32','original-m16-repeat']
FILES={f'layer{i}.nb' for i in range(0,24,2)}|{f'layer{i}.nb' for i in range(24,28)}|{f'prefill-layer{i}.nb' for i in range(0,24,2)}|{'embeddings.f32'}
TAIL={f'prefill-layer{i}.nb' for i in range(0,24,2)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def flags(remote):
 return {'OPI_TTFT_TAIL_MIN':'8','OPI_GROUP_PREFILL':'1','OPI_MASK_TEMPLATE':'1','OPI_GREEDY_I16_SINGLE_PASS':'1','OPI_PREFILL8_DIR':remote+'/m8'}
def expected_links(remote):
 links={}
 for bank in ['m16','m32']:
  x={f'layer{i}.nb':remote+f'/artifacts/d-l{i}.nb' for i in range(0,24,2)}
  x.update({f'prefill-layer{i}.nb':remote+f'/artifacts/{bank}-l{i}.nb' for i in range(0,24,2)})
  x.update({f'layer{i}.nb':BASE+f'/layer{i}.nb' for i in range(24,28)})
  x['embeddings.f32']=BASE+'/embeddings.f32';links[bank]=x
 links['m8']={f'prefill-layer{i}.nb':remote+f'/artifacts/m8-l{i}.nb' for i in range(0,24,2)}
 return links

def validate_aba(cfg):
 remote=cfg['remote'];assert remote.startswith('/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/') and '..' not in Path(remote).parts
 assert re.fullmatch('[A-Za-z0-9_-]+',Path(remote).name)
 assert cfg['repeats']==1 and cfg['quality'] is True and cfg['diagnostic'] is True
 assert len(cfg['feedback_tokens'])==32 and cfg.get('extra_diagnostic_case_ids',[])==[]
 assert cfg['defaults']['capacity']==512 and cfg['defaults']['layers']==24 and cfg['defaults']['vocab']==151674
 assert cfg['overlay_contract']==expected_links(remote)
 artifacts=cfg['artifacts'];assert set(artifacts)=={f'{k}-l{i}.nb' for k in ['d','m8','m16','m32'] for i in range(0,24,2)}
 for n,a in artifacts.items():
  assert a['precision']=='original' and a['bytes']>0 and re.fullmatch('[0-9a-f]{64}',a['sha256'])
 assert [v['name'] for v in cfg['variants']]==NAMES
 def common(v):return {k:x for k,x in v.items() if k not in ['name','graph','expected_graph_sha256','prefill_batch']}
 a,b,c=cfg['variants'];assert common(a)==common(b)==common(c)
 assert {k:v for k,v in a.items() if k!='name'}=={k:v for k,v in c.items() if k!='name'}
 refs=cfg['head_embedding_hashes'];assert set(refs)=={f'layer{i}.nb' for i in range(24,28)}|{'embeddings.f32'}
 for v,bank,m in zip(cfg['variants'],['m16','m32','m16'],[16,32,16]):
  assert v['graph']==remote+'/'+bank and v['prefill_batch']==m and v['capacity']==512
  assert v['binary']==remote+'/resident-cap512' and v['expected_binary_sha256']==BINARY_SHA
  assert v['env']==flags(remote) and v['speed_profileflag']=='none'
  hashes={remote+'/'+bank+'/'+n:(refs[n] if target.startswith(BASE+'/') else artifacts[Path(target).name]['sha256']) for n,target in cfg['overlay_contract'][bank].items()}
  assert v['expected_graph_sha256']==hashes
  assert v['expected_tail_sha256']=={remote+'/m8/'+n:artifacts[Path(t).name]['sha256'] for n,t in cfg['overlay_contract']['m8'].items()}
 assert cfg['fixture_path']==remote+'/inputs.json' and cfg['rag_catalog']==remote+'/atoms.json'
 return True

def freeze(raw,remote):
 import copy
 cfg=copy.deepcopy(raw);cfg['remote']=remote;cfg['defaults']['capacity']=512
 cfg['overlay_contract']=expected_links(remote);cfg['fixture_path']=remote+'/inputs.json';cfg['rag_catalog']=remote+'/atoms.json'
 refs={Path(k).name:v for k,v in raw['variants'][0]['expected_graph_sha256'].items()}
 cfg['head_embedding_hashes']={n:refs[n] for n in refs if n=='embeddings.f32' or n in {f'layer{i}.nb' for i in range(24,28)}}
 cfg.pop('candidate',None);cfg['variants']=[]
 for name,bank,m in zip(NAMES,['m16','m32','m16'],[16,32,16]):
  expected={remote+'/'+bank+'/'+n:(cfg['head_embedding_hashes'][n] if t.startswith(BASE+'/') else cfg['artifacts'][Path(t).name]['sha256']) for n,t in cfg['overlay_contract'][bank].items()}
  tail={remote+'/m8/'+n:cfg['artifacts'][Path(t).name]['sha256'] for n,t in cfg['overlay_contract']['m8'].items()}
  cfg['variants'].append({'name':name,'graph':remote+'/'+bank,'capacity':512,'prefill_batch':m,'binary':remote+'/resident-cap512','expected_binary_sha256':BINARY_SHA,'env':flags(remote),'speed_profileflag':'none','expected_graph_sha256':expected,'expected_tail_sha256':tail})
 validate_aba(cfg);return cfg

def verify_overlay(cfg):
 validate_aba(cfg);root=Path(cfg['remote']);evidence=[]
 assert {f.name for f in (root/'artifacts').iterdir()}==set(cfg['artifacts'])
 for n,a in cfg['artifacts'].items():
  f=root/'artifacts'/n;assert f.is_file() and not f.is_symlink() and f.stat().st_size==a['bytes'] and sha(f)==a['sha256']
 for bank,links in cfg['overlay_contract'].items():
  d=root/bank;assert d.is_dir() and not d.is_symlink() and {x.name for x in d.iterdir()}==set(links)
  for n,target in links.items():
   f=d/n;assert f.is_symlink() and os.readlink(f)==target and f.resolve(strict=True)==Path(target).resolve(strict=True)
   evidence.append({'bank':bank,'name':n,'target':target,'bytes':f.stat().st_size})
 return {'scope':'48 exact original precision artifacts SHA verified; 29+29 main and12 tail symlinks; head and embedding SHA checked before worker load','links':evidence}
