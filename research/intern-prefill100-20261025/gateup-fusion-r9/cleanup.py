from pathlib import Path
import json,hashlib,sys,shlex
D=Path(__file__).resolve().parent;T=D.parent;sys.path.insert(0,str(T));from controller_down_outputfp16_tmpfs import b,locked
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=json.loads((D/'controller.json').read_text());assert state['status']=='complete' and state['returncode']==0
remote='/tmp/opi-gateup-fusion-r9-a2';assert state['remote']==remote
expected=json.loads((D/'deployment-sha.json').read_text())
for p in (D/'results').rglob('*'):
 if p.is_file():expected[str(p.relative_to(D))]=sha(p)
for n in ['process.log','exit']:expected[n]=sha(D/n)
code="""from pathlib import Path
import hashlib,json,re,shutil
r=Path(REMOTE);expected=json.loads(EXPECTED)
assert r.is_dir() and not r.is_symlink()
files={}
for p in r.rglob('*'):
 assert not p.is_symlink()
 if p.is_file():files[str(p.relative_to(r))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
assert set(files)==set(expected)
assert all(files[n]['sha256']==h for n,h in expected.items())
driver=Path('/sys/kernel/debug/viplite/mem_profile').read_text();assert int(re.findall(r'0x([0-9a-fA-F]+)',driver)[0],16)==0
result={'remote':str(r),'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'all_sha_equal':True,'driver0_before':True,'sha256':expected}
shutil.rmtree(r);result['removed']=not r.exists();print(json.dumps(result))
""".replace('REMOTE',repr(remote)).replace('EXPECTED',repr(json.dumps(expected)))
with b.hostlock():
 z=locked('python3 -c '+shlex.quote(code),capture_output=True,text=True,check=True,timeout=45)
 receipt=json.loads(z.stdout);(D/'BOARD_CLEANUP.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'}))
