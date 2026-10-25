"""Host and board structural contract. No network calls."""
from pathlib import Path
import copy,hashlib,os,re
BASE='/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/c512-m16-m32-aba-r1/m32'
TAIL='/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/c512-m16-m32-aba-r1/m8'
BINARY_SHA='bf19922fb62bdde70a15ccde742abf9b4ffa6e5b116e73929fa7310442dc1e11'
NAMES=['c512-m32-original','c512-m32-conv-l2','c512-m32-repeat']
FLAGS={'OPI_TTFT_TAIL_MIN':'8','OPI_GROUP_PREFILL':'1','OPI_MASK_TEMPLATE':'1','OPI_GREEDY_I16_SINGLE_PASS':'1','OPI_PREFILL8_DIR':TAIL}
def flags(remote):return dict(FLAGS)
FILES={f'layer{i}.nb' for i in range(0,24,2)}|{f'layer{i}.nb' for i in range(24,28)}|{f'prefill-layer{i}.nb' for i in range(0,24,2)}|{'embeddings.f32'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def valid_sha(x):return isinstance(x,str) and bool(re.fullmatch('[0-9a-f]{64}',x))
def validate_candidate(c,local=False):
 assert c['replace_only']=='prefill-layer2.nb'
 assert [c[k] for k in ['capacity','query_tokens','fusion_width','first_layer']]==[512,32,2,2]
 assert valid_sha(c['candidate_sha256']) and valid_sha(c['baseline_nb_sha256'])
 assert c['candidate_sha256']!=c['baseline_nb_sha256'] and type(c['candidate_bytes']) is int and 0<c['candidate_bytes']<100*1024**2
 assert all(c[k] is True for k in ['same_quantizer_bytes','same_14_fc_exported_weight_bytes','same_external_port_contract'])
 if local:
  f=Path(c['candidate_nb']);assert f.is_file() and sha(f)==c['candidate_sha256'];assert f.stat().st_size==c['candidate_bytes']
def validate_aba(cfg):
 validate_candidate(cfg['candidate'])
 assert cfg['repeats']==1 and cfg['quality'] is True and cfg['diagnostic'] is True
 assert [v['name'] for v in cfg['variants']]==NAMES
 assert cfg['defaults']['capacity']==512 and cfg['defaults']['layers']==24 and cfg['defaults']['prefill_batch']==32
 a,b,c=cfg['variants'];assert a['graph']==c['graph']==BASE
 assert a['expected_graph_sha256']==c['expected_graph_sha256']
 common=lambda v:{k:x for k,x in v.items() if k not in ['name','graph','expected_graph_sha256']}
 assert common(a)==common(b)==common(c), 'all non-graph variant settings must be equal'
 ref={Path(k).name:v for k,v in a['expected_graph_sha256'].items()}
 assert set(ref)==FILES and len(a['expected_graph_sha256'])==29
 assert all(str(Path(BASE)/n) in a['expected_graph_sha256'] for n in FILES)
 assert ref['prefill-layer2.nb']==cfg['candidate']['baseline_nb_sha256']
 assert all(valid_sha(v) for v in ref.values())
 for v in cfg['variants']:
  assert v['prefill_batch']==32 and v['expected_tail_sha256']==cfg['tail_sha256']
  assert set(cfg['tail_sha256'])=={TAIL+f'/prefill-layer{i}.nb' for i in range(0,24,2)}
  assert v['env']==FLAGS and v['capacity']==512 and v['expected_binary_sha256']==BINARY_SHA
  assert v['binary']==a['binary']==cfg['remote']+'/resident-cap512' and v['speed_profileflag']=='none'
 assert b['graph']!=BASE and b['graph']==cfg['overlay_contract']['overlay']
 alt={Path(k).name:v for k,v in b['expected_graph_sha256'].items()}
 assert set(alt)==FILES and len(b['expected_graph_sha256'])==29
 assert all(str(Path(b['graph'])/n) in b['expected_graph_sha256'] for n in FILES)
 assert [n for n in sorted(FILES) if alt[n]!=ref[n]]==['prefill-layer2.nb']
 assert alt['prefill-layer2.nb']==cfg['candidate']['candidate_sha256']
 assert len(cfg['feedback_tokens'])==32 and cfg.get('extra_diagnostic_case_ids',[])==[]
 oc=cfg['overlay_contract'];assert set(oc['symlinks'])==FILES
 for n in FILES:assert oc['symlinks'][n]==(oc['candidate_remote'] if n=='prefill-layer2.nb' else str(Path(BASE)/n))
 assert Path(oc['candidate_remote']).parent==Path(b['graph']).parent
 assert Path(oc['candidate_remote']).name=='prefill-layer2-conv.nb'
 return True

def freeze(cfg,remote):
 cfg=copy.deepcopy(cfg);validate_candidate(cfg['candidate'],local=True);cfg['remote']=remote
 assert remote.startswith('/opt/opi-vlm/intern-prefill100-20261025/capacity512-conv-full/runs/') and '..' not in Path(remote).parts
 overlay=remote+'/overlay';candidate=remote+'/prefill-layer2-conv.nb'
 ref={Path(k).name:v for k,v in cfg['variants'][0]['expected_graph_sha256'].items()}
 for v in cfg['variants']:v['binary']=remote+'/resident-cap512'
 v=cfg['variants'][1];v['graph']=overlay
 v['expected_graph_sha256']={str(Path(overlay)/n):(cfg['candidate']['candidate_sha256'] if n=='prefill-layer2.nb' else h) for n,h in ref.items()}
 cfg['fixture_path']=remote+'/inputs.json';cfg['rag_catalog']=remote+'/atoms.json'
 cfg['overlay_contract']={'overlay':overlay,'candidate_remote':candidate,'symlinks':{n:(candidate if n=='prefill-layer2.nb' else str(Path(BASE)/n)) for n in sorted(FILES)}}
 validate_aba(cfg);return cfg

def verify_overlay(cfg):
 validate_aba(cfg);oc=cfg['overlay_contract'];directory=Path(oc['overlay']);candidate=Path(oc['candidate_remote'])
 assert directory.is_dir() and not directory.is_symlink()
 assert candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size==cfg['candidate']['candidate_bytes']
 assert sha(candidate)==cfg['candidate']['candidate_sha256']
 assert {x.name for x in directory.iterdir()}==FILES
 records=[]
 for name,target in sorted(oc['symlinks'].items()):
  link=directory/name;assert link.is_symlink() and os.readlink(link)==target
  resolved=link.resolve(strict=True);assert resolved==Path(target).resolve(strict=True) and resolved.is_file()
  records.append({'name':name,'link_target':target,'resolved_path':str(resolved),'bytes':resolved.stat().st_size})
 return {'scope':'exact29 symlinks; only prefill-layer2 points to new candidate; full SHA checked by benchmark','files':records}
