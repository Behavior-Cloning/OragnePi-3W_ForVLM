"""Board-only fixed fixture benchmark. Caller owns hostlock + board flock."""
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
    assert len(doc['performance']) == 12 and len(doc['quality']) == 12
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
        assert len(case['tokens']) == case['static_tokens'] + case['body_tokens'] + 8
        assert case['expected_reused_tokens'] == case['static_tokens']
        assert case['expected_new_tokens'] == case['body_tokens'] + 8
    for case in doc['performance']:
        key = (case['static_tokens'], case['body_tokens'])
        cells.setdefault(key, []).append(case)
    assert set(cells) == {(64,32),(64,64),(128,32),(128,64)}
    assert all(len(v) == 3 for v in cells.values())
    return [v[0] for v in cells.values()]

def decode_segments(tokens):
    # Token events contain runtime elapsed seconds; exclude first token from decode.
    intervals = [tokens[i]['elapsed']-tokens[i-1]['elapsed'] for i in range(1,len(tokens))]
    result = []
    for lo,hi in [(2,8),(9,16),(17,24),(25,32)]:
        xs = intervals[lo-2:hi-1]
        result.append({'output_positions':[lo,hi], 'intervals':len(xs),
                       'seconds':sum(xs), 'tokens_s':len(xs)/sum(xs) if xs and sum(xs)>0 else None})
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int);ap.add_argument('--phases',default=None)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    repeats=args.repeats if args.repeats is not None else cfg.get('repeats',3)
    phases=set(args.phases.split(',')) if args.phases else {'performance'} | ({'quality'} if cfg.get('quality',True) else set()) | ({'diagnostic'} if cfg.get('diagnostic',True) else set())
    assert phases and phases <= {'performance','quality','diagnostic'} and repeats>=1
    assert len({v['name'] for v in cfg['variants']})==len(cfg['variants'])
    for v in cfg['variants']:
        assert v['name'] and '/' not in v['name'] and v['name'] not in {'.','..'}
        env=v.get('env',{})
        assert not any(k in env for k in ['OPI_SESSION_FD','OPI_SESSION_DUMP_DIR'])
        if 'performance' in phases: assert not env.get('OPI_PROFILE_NPU'), 'NPU queries forbidden in timed run'
        if v.get('speed_profileflag')=='npu_query': assert phases=={'diagnostic'}
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    base=Path(cfg['base_path'])
    sys.path[:0]=[str(ROOT),str(base),str(base/'deps'),'/opt/opi-vlm/audio-lab-20261018/deps']
    from agent_replay import Frontend,wav,snapshot,np
    from resident_worker import ResidentWorker
    from context_rag import Retriever
    from tokenizers import Tokenizer
    fixture_path=resolve(cfg.get('fixture_path','aligned-inputs.json'));doc=json.loads(fixture_path.read_text())
    tokenizer_path=Path(cfg['tokenizer_path']);assert sha(tokenizer_path)==doc['tokenizer_sha256']
    tok=Tokenizer.from_file(str(tokenizer_path));encode=lambda s:tok.encode(s,add_special_tokens=False).ids
    diag_cases=validate_fixture(doc,encode)
    feedback=cfg.get('feedback_tokens',[100662]*32)
    assert len(feedback)==32 and all(type(t) is int and 0<=t<cfg['defaults']['vocab'] for t in feedback)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    (out/'fixtures.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    manifest={n:sha(ROOT/n) for n in ['board_benchmark.py','resident_worker.py','context_rag.py']}
    manifest.update(fixture_sha256=sha(fixture_path),tokenizer_sha256=sha(tokenizer_path),frontend_sha256=sha(base/'agent_replay.py'))
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2))
    os.sched_setaffinity(0,{7})
    log=(out/'events.jsonl').open('w',buffering=1);lock=threading.Lock();stop=threading.Event()
    state={'worker':None,'variant':None,'phase':'startup'};background_errors=[];threads=[]
    refs={};diag_refs={};graph_hash_cache={};graph_verifications=[];front=None;success=False
    def emit(kind,**kw):
        with lock: log.write(json.dumps({'event':kind,'at':time.monotonic(),**kw},ensure_ascii=False)+'\n')
    def audio_state():
        return {'frames':front.frames,'overruns':front.overruns,'max_lag_s':front.max_lag,'max_compute_s':front.max_compute,'stage_seconds':dict(front.stage_seconds),'queue':front.jobs.qsize(),'error':front.error}
    def check_health():
        if background_errors or front.error:raise RuntimeError(str(background_errors or front.error))
        if int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())>=78000:raise RuntimeError('78C guard')
    def feed():
        try:
            i=0
            while not stop.is_set():
                if front.error:raise RuntimeError(str(front.error))
                if front.jobs.empty():
                    i+=1;x=wav(base/'speech'/('%02d.wav'%((i-1)%12+1)))
                    front.jobs.put({'id':i,'kind':'command','mode':'stream','audio':np.concatenate([x,np.zeros(32000,np.float32)])})
                try:
                    while True:
                        e=front.events.get_nowait();e.pop('audio',None)
                        if e.get('event')=='endpoint':emit('audio_endpoint',text=e.get('text'))
                except queue.Empty:pass
                stop.wait(.1)
        except BaseException:
            background_errors.append(traceback.format_exc());emit('audio_error',error=background_errors[-1]);stop.set()
    def monitor():
        try:
            with (out/'resources.jsonl').open('w',buffering=1) as f:
                while not stop.wait(.5):
                    worker=state['worker'];ids=[os.getpid()]+([worker.process.pid] if worker else [])
                    row=snapshot(ids);row['frontend']=audio_state();row['variant']=state['variant'];row['phase']=state['phase']
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    if row['temp_mC']>=78000:raise RuntimeError('78C guard')
        except BaseException:
            background_errors.append(traceback.format_exc());emit('monitor_error',error=background_errors[-1]);stop.set()
    def request(worker,ids,n,**kwargs):
        start=time.monotonic();tokens=[];done=None
        for item in worker.generate(ids,n,**kwargs):
            if item['event']=='token':tokens.append({**item,'observed_s':time.monotonic()-start})
            else:done=item
        assert done is not None and len(tokens)==done['generated_tokens']
        return done,tokens,time.monotonic()-start
    def parity(key,ids):
        if key not in refs:
            refs[key]=(state['variant'],ids);return {'baseline_variant':state['variant'],'reference':True,'exact':True}
        name,prior=refs[key]
        return {'baseline_variant':name,'reference':False,'exact':prior==ids,'shared_positions':min(len(prior),len(ids)),
                'matching_positions':sum(a==b for a,b in zip(prior,ids)),'reference_tokens':len(prior)}
    try:
        front=Frontend('stream',.5);front.start()
        assert front.ready.wait(120) and not front.error
        rag=Retriever(catalog_path=Path(cfg['rag_catalog']))
        emit('resident_models',rag=rag.metadata,phases=sorted(phases),repeats=repeats,audio_policy='Frontend streamed prerecorded WAV; at most one queued whole utterance; not microphone')
        threads=[threading.Thread(target=fn,daemon=True) for fn in [feed,monitor]]
        for th in threads:th.start()
        for variant in cfg['variants']:
            check_health();name=variant['name'];settings={**cfg['defaults'],**variant};state.update(variant=name,phase='load')
            graph=Path(settings['graph']);binary=Path(settings['binary']);env=settings.get('env',{})
            state['phase']='verify_graph'
            verified=verify_graph_files(settings.get('expected_graph_sha256',{}),graph,graph_hash_cache)
            graph_verifications.append({'variant':name,'files':verified,'coverage':'only explicitly declared files; absent mapping is not full graph verification'})
            (out/'graph-provenance.json').write_text(json.dumps(graph_verifications,ensure_ascii=False,indent=2))
            emit('graph_verified',variant=name,files=verified)
            state['phase']='load'
            assert max(len(c['tokens'])+31 for c in doc['performance']+doc['quality'])<=settings['capacity']
            command=['env',*[k+'='+str(v) for k,v in env.items()],'taskset','-c',settings['affinity'],str(binary),str(graph),str(settings['layers']),str(settings['capacity']),str(settings['vocab']),str(settings.get('embedding',graph/'embeddings.f32')),str(settings['prefill_batch']),str(settings['eos'])]
            dumps=out/(name+'-diagnostic');dumps.mkdir()
            worker=ResidentWorker(command,base,out/(name+'.log'),dump_dir=dumps);state['worker']=worker
            assert worker.capacity==settings['capacity'] and worker.ready.get('layers')==settings['layers']
            emit('loaded',variant=name,ready=worker.ready,binary_sha256=sha(binary),settings=settings,
                 graph_files=[{'name':z.name,'bytes':z.stat().st_size} for z in sorted(graph.glob('*.nb'))])
            # All measured requests reset and prepare only the frozen system prefix.
            warm=doc['performance'][0];request(worker,warm['prefix_tokens'],1,reset=True,diagnostic=False)
            request(worker,warm['tokens'],32 if 'performance' in phases else 1,feedback=feedback if 'performance' in phases else None,diagnostic=False)
            def prepare(case):
                check_health();prep,_,_=request(worker,case['prefix_tokens'],1,reset=True,diagnostic=False)
                assert prep['reused_tokens']==0 and not prep['diagnostic']
                return prep
            def validate(done,case,diagnostic,fixed,n):
                assert done['reused_tokens']==case['expected_reused_tokens']
                assert done['new_tokens']==case['expected_new_tokens']
                assert done['diagnostic']==diagnostic and done['fixed_feedback']==fixed
                if fixed:assert done['generated_tokens']==n
            if 'performance' in phases:
                state['phase']='performance'
                for repeat in range(repeats):
                    cases=doc['performance'] if repeat%2==0 else list(reversed(doc['performance']))
                    for case in cases:
                        prep=prepare(case);done,tokens,wall=request(worker,case['tokens'],32,feedback=feedback,diagnostic=False,timeout=90)
                        validate(done,case,False,True,32)
                        assert not list(dumps.glob('*')), 'Unexpected dump during speed phase'
                        emit('measure',variant=name,repeat=repeat,case=case['id'],static_tokens=case['static_tokens'],system_body_tokens=case['system_body_tokens'],body_tokens=case['body_tokens'],template_overhead=case['template_overhead'],prompt_sha256=case['tokens_sha256'],prefix_prepare_metrics=prep,metrics=done,wall_s=wall,token_events=tokens,decode_segments=decode_segments(tokens),feedback_tokens=feedback,output_parity=parity(('performance',repeat,case['id']),done['tokens']),speed_profileflag=settings.get('speed_profileflag','none'))
            if 'quality' in phases:
                state['phase']='quality'
                for case in doc['quality']:
                    prep=prepare(case);done,tokens,wall=request(worker,case['tokens'],32,diagnostic=False,timeout=90)
                    validate(done,case,False,False,32);reply=tok.decode(done['tokens']);t=time.monotonic();actions=rag.query(reply,k=3);rag_s=time.monotonic()-t
                    assert not list(dumps.glob('*')), 'Unexpected dump during natural phase'
                    emit('quality',variant=name,case=case['id'],static_tokens=case['static_tokens'],system_body_tokens=case['system_body_tokens'],system_template_tokens=case['static_tokens']-case['system_body_tokens'],body_tokens=case['body_tokens'],template_overhead=case['template_overhead'],dynamic_tokens=len(case['tokens'])-case['static_tokens'],total_prompt_tokens=len(case['tokens']),expected_new_tokens=case['expected_new_tokens'],actual_new_tokens=done['new_tokens'],actual_reused_tokens=done['reused_tokens'],output_tokens_including_eos=done['generated_tokens'],output_tokens_excluding_terminal_eos=done['generated_tokens']-int(done['tokens'][-1]==settings['eos']),input=case['input'],reply=reply,metrics=done,wall_s=wall,token_events=tokens,prefix_prepare_metrics=prep,top3=actions,rag_s=rag_s,rag_metrics=rag.last_query,output_parity=parity(('quality',case['id']),done['tokens']))
                # Extra natural-decoding coverage of all four exact-length tail8 cells.
                # Cost padding is retained and these are not independent quality examples.
                for case in diag_cases:
                    prep=prepare(case);done,tokens,wall=request(worker,case['tokens'],32,diagnostic=False,timeout=90)
                    validate(done,case,False,False,32)
                    if settings['env'].get('OPI_PREFILL8_DIR'):
                        assert done.get('true_tail8_calls')==1, done
                    assert not list(dumps.glob('*')), 'Unexpected dump during natural cost coverage'
                    emit('quality_cost',variant=name,case=case['id'],static_tokens=case['static_tokens'],system_body_tokens=case['system_body_tokens'],body_tokens=case['body_tokens'],template_overhead=case['template_overhead'],cost_padding=True,input=case['body'],reply=tok.decode(done['tokens']),metrics=done,wall_s=wall,token_events=tokens,prefix_prepare_metrics=prep,output_parity=parity(('quality_cost',case['id']),done['tokens']))
            if 'diagnostic' in phases:
                state['phase']='diagnostic'
                for case in diag_cases:
                    prepare(case);done,_,_=request(worker,case['tokens'],4,feedback=feedback[:4],diagnostic=True,timeout=90)
                    validate(done,case,True,True,4);file=dumps/(str(done['id'])+'-logits.f32');raw=file.read_bytes()
                    arr=np.frombuffer(raw,dtype='<f4');assert len(raw)==4*settings['vocab']*4 and np.isfinite(arr).all()
                    compare={'reference':True,'baseline_variant':name}
                    if case['id'] in diag_refs:
                        refname,refpath=diag_refs[case['id']];ref=np.fromfile(refpath,dtype='<f4')
                        assert len(ref)==len(arr)
                        delta=arr.astype(np.float64)-ref;denom=float(np.linalg.norm(ref.astype(np.float64)))
                        compare={'reference':False,'baseline_variant':refname,'relative_l2':float(np.linalg.norm(delta))/max(denom,1e-30),'max_abs':float(np.abs(delta).max()),'top1_matches':int(np.sum(arr.reshape(4,-1).argmax(1)==ref.reshape(4,-1).argmax(1))),'steps':4}
                    else:diag_refs[case['id']]=(name,file)
                    emit('diagnostic',variant=name,case=case['id'],metrics=done,logits_file=str(file.relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),logits_parity=compare,timing_eligible=False)
            state['phase']='close';closed=worker.close();state['worker']=None
            emit('closed',variant=name,metrics=closed);assert closed.get('driver_bytes')==0
        check_health();success=True
    except BaseException:
        emit('error',traceback=traceback.format_exc());raise
    finally:
        worker=state['worker']
        if worker:
            try:
                closed=worker.close();emit('closed_on_failure',variant=state['variant'],metrics=closed)
            except BaseException:
                emit('release_error',variant=state['variant'],traceback=traceback.format_exc())
                try:worker.abort()
                except BaseException:pass
            state['worker']=None
        stop.set()
        if front:front.stop.set();front.join(15)
        for th in threads:th.join(5)
        release={'status':'complete' if success and not background_errors and front and not front.is_alive() else 'failed','audio':audio_state() if front else None,'background_errors':background_errors,'scope':'Intern NPU text decoder + resident synthetic speech frontend and RAG; no image encoder, microphone or physical motors','timing': 'fixed feedback32, diagnostic=False; natural quality and diagnostic4 separated','parity':'exact tokens/logits versus first configured runtime; not an FP32 accuracy oracle'}
        (out/'meta.json').write_text(json.dumps(release,ensure_ascii=False,indent=2));emit('complete' if release['status']=='complete' else 'failed',**release);log.close()

if __name__=='__main__':main()
