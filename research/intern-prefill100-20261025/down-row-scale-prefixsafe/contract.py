"""Frozen C512 M32 A/B/A. Only prefill-layer2.nb may differ; no extra KV reuse."""
from pathlib import Path
import hashlib,json,os,copy
ROOT=Path(__file__).resolve().parent
REMOTE='/dev/shm/opi-prefill100-down-row-scale-prefixsafe-r1'
BINARY_SHA='0ed64448e560cd3b973491cc4bcd2901e27818ff45fb165659d95c5e5d5fdb67'
NB_SHA='f0bebd8b9489f487a5b72658184c9d8f90bb8ec36a6682afde9bb61e097ea25a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def baseline():return json.loads((ROOT/'baseline-config.json').read_text())
def flags(remote, protected=False):
 assert remote==REMOTE
 v=baseline()['variants'][1];e=copy.deepcopy(v['env'])
 if protected:
  e['OPI_FIRST_PREFILL_L2']=v['graph']+'/prefill-layer2.nb'
  e['OPI_FIRST_PREFILL8_L2']=e['OPI_PREFILL8_DIR']+'/prefill-layer2.nb'
 return e
def validate_aba(c):
 b=baseline();v=b['variants'][1]
 assert sha(ROOT/'baseline-config.json')==c['baseline_config_sha256']
 assert c['remote']==REMOTE and c['repeats']==1 and c['quality'] is True and c['diagnostic'] is True
 for k in ['defaults','feedback_tokens','base_path','tokenizer_path','expected_fixture_sha256']:assert c[k]==b[k],k
 assert c['fixture_path']==REMOTE+'/inputs.json' and c['rag_catalog']==REMOTE+'/atoms.json'
 assert c['candidate']['sha256']==NB_SHA and c['candidate']['bytes']==29765440
 a=json.loads((ROOT/'static-audit.json').read_text())
 assert sha(ROOT/'static-audit.json')==c['candidate']['static_audit_sha256'] and a['static_file_audit_pass']
 for key in ['graph','quant']:assert a['candidate_'+key+'_sha256']==c['candidate'][key+'_sha256']
 assert a['candidate_nb_sha256']==NB_SHA
 assert [x['name'] for x in c['variants']]==['original-m32','row-scale-m32','protected-system','original-m32-repeat']
 for i,w in enumerate(c['variants']):
  expected=copy.deepcopy(v);expected.update(name=w['name'],binary=REMOTE+'/resident-prefixsafe512',expected_binary_sha256=BINARY_SHA,protect_system=(i==2),env=flags(REMOTE,i==2))
  if i in (1,2):
   expected['graph']=REMOTE+'/candidate'
   expected['expected_graph_sha256']={REMOTE+'/candidate/'+Path(k).name:(NB_SHA if Path(k).name=='prefill-layer2.nb' else h) for k,h in v['expected_graph_sha256'].items()}
  assert w==expected,w['name']
 assert c['overlay_contract']=={Path(k).name:(REMOTE+'/candidate.nb' if Path(k).name=='prefill-layer2.nb' else k) for k in v['expected_graph_sha256']}
 return True
def verify_overlay(c):
 validate_aba(c);nb=Path(REMOTE)/'candidate.nb'
 assert not nb.is_symlink() and nb.stat().st_size==c['candidate']['bytes'] and sha(nb)==NB_SHA
 folder=Path(REMOTE)/'candidate';assert folder.is_dir() and not folder.is_symlink()
 assert {x.name for x in folder.iterdir()}==set(c['overlay_contract'])
 evidence=[]
 for n,t in c['overlay_contract'].items():
  f=folder/n;assert f.is_symlink() and os.readlink(f)==t and f.resolve(strict=True)==Path(t).resolve(strict=True)
  evidence.append({'name':n,'target':t,'bytes':f.stat().st_size})
 return {'scope':'one changed M32 prefill block; candidate SHA verified; all original main+tail hashes checked by runner before model load','links':evidence}
