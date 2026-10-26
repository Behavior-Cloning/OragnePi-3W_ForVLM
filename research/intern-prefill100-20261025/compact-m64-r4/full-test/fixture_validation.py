"""C1280 full-model baseline, core24 x repeat2, not a new speedup. Caller owns both locks."""
import argparse, hashlib, json, math, os, queue, struct, sys, threading, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stream_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk=stream.read(4*1024*1024)
            if not chunk:break
            digest.update(chunk)
    return digest.hexdigest()

def verify_graph_files(expected,graph,cache):
    records=[]
    for declared,want in sorted(expected.items()):
        assert isinstance(want,str) and len(want)==64 and all(c in '0123456789abcdefABCDEF' for c in want)
        path=resolve(declared,graph).resolve(strict=True);before=path.stat()
        key=(str(path),before.st_size,before.st_mtime_ns)
        reused=key in cache
        got=cache.get(key)
        if got is None:
            got=stream_sha(path);after=path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), 'graph changed during hashing'
            cache[key]=got
        records.append({'declared_path':declared,'resolved_path':str(path),'bytes':before.st_size,'expected_sha256':want.lower(),'sha256':got,'cache_reused':reused})
        assert got==want.lower(), 'graph SHA mismatch: '+str(path)
    return records

def token_hash(ids): return hashlib.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids)).hexdigest()
def resolve(path, base=ROOT):
    p = Path(path)
    return p if p.is_absolute() else base / p

def attach_system_metadata(doc):
    systems={system['prefix_sha256']:system for system in doc['systems']}
    assert len(systems)==len(doc['systems'])
    for case in doc['performance']+doc['quality']:
        system=systems[case['prefix_sha256']]
        assert case['prefix_tokens']==system['prefix_tokens']
        assert case['static_tokens']==len(system['prefix_tokens'])
        body=system['system_body_tokens'];template=case['static_tokens']-body
        assert template==5
        for key,value in [('system_body_tokens',body),('system_template_tokens',template)]:
            assert key not in case or case[key]==value, (case['id'],key)
            case[key]=value

def validate_fixture(doc, encode):
    assert len(doc['performance']) == 39 and len(doc['quality']) == 12
    attach_system_metadata(doc)
    cells = {}
    for system in doc['systems']:
        assert encode(system['prefix']) == system['prefix_tokens']
        assert len(encode(system['system_body'])) == system['system_body_tokens']
        assert token_hash(system['prefix_tokens']) == system['prefix_sha256']
    for case in doc['performance'] + doc['quality']:
        assert encode(case['prompt_text']) == case['tokens']
        assert token_hash(case['tokens']) == case['tokens_sha256']
        assert token_hash(case['prefix_tokens']) == case['prefix_sha256']
        assert len(case['prefix_tokens']) == case['static_tokens']
        assert case['tokens'][:case['static_tokens']] == case['prefix_tokens']
        assert len(encode(case.get('body',case.get('input')))) == case['body_tokens']
        assert case['template_overhead'] == 8 and case['prompt_tokens'] == len(case['tokens'])
        assert len(case['tokens']) == case['static_tokens'] + case['body_tokens'] + 8
        assert case['expected_reused_tokens'] == case['static_tokens']
        assert case['expected_new_tokens'] == case['body_tokens'] + 8
    for case in doc['performance']:
        key = (case['static_tokens'], case['body_tokens'])
        cells.setdefault(key, []).append(case)
    assert set(cells) == ({(s,d) for s in (64,128) for d in (64,128,192,256)} | {(32,d) for d in (32,64,128,256)} | {(64,32)})
    assert len({c['id'] for c in doc['performance']+doc['quality']})==51
    assert all(len(v) == 3 for v in cells.values())
    return [v[0] for v in cells.values()]

