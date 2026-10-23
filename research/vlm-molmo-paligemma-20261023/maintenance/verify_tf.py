from pathlib import Path
import hashlib,json,time
p=Path(__file__).resolve().parent;f=p/'tf-compression.json';r=json.loads(f.read_text());h=hashlib.sha256();start=time.time()
with Path('/mnt/f/Files/OPi-Lab/backups/opi-tf-20261017-original.img').open('rb') as inp:
 for b in iter(lambda:inp.read(16*1024*1024),b''):h.update(b)
r.update(actual_sha256=h.hexdigest(),matches_original=h.hexdigest()==r['expected_sha256'],verify_seconds=time.time()-start,status='compression_failed_original_verified' if h.hexdigest()==r['expected_sha256'] else 'HASH_MISMATCH');f.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
