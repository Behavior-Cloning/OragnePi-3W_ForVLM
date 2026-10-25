"""Analyze recovered compact-row two-layer data only, no board access."""
from pathlib import Path
import hashlib,json,tarfile,numpy as np
T=Path(__file__).resolve().parents[2];run='compact-row-down-r1';R=T/'results'/run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ctl=read(T/(run+'-controller.json'));assert ctl['status']=='complete' and ctl['returncode']==0
assert read(R/'meta.json')['status']=='complete'
a=T/(run+'.tar.gz');assert sha(a)==ctl['archive_sha256']
with tarfile.open(a) as tf:
 members=tf.getmembers();names=[m.name for m in members];assert len(names)==len(set(names)), "Duplicate archive members"
 for m in members:
  f=(R/m.name).resolve();assert f==R.resolve() or R.resolve() in f.parents
  assert m.isfile() or m.isdir()
  if m.isfile():assert f.stat().st_size==m.size and hashlib.sha256(tf.extractfile(m).read()).hexdigest()==sha(f)
cfg=read(R/'config.json');assert cfg==read(T/(run+'-config.json'))
for n,h in {**cfg['source_sha256'],**cfg['fixture_metadata_sha256']}.items():assert sha(T/n)==h,n
assert sha(Path(cfg['candidate_status_path']))==cfg['candidate_status_sha256']
e=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in e]==['start','result']*6+['complete'];ev={x['case']:x for x in e if x['event']=='result'}
expected_labels=[f'm{M}-{phase}' for M in [64,256] for phase in ['before','row','after']]
assert [x['case'] for x in e if x['event']=='result']==expected_labels, 'Result order or duplicate labels'
assert len(ev)==6
rows=[];segments=[];boundary_temps=[];out_arrays={}
for M in [64,256]:
 meta=read(T/f'fixtures-compact/c128-m{M}-l2/probe-fixture.json');expected=sum(n for _,n in meta['outputs']);ports0=None
 for phase in ['before','row','after']:
  label=f'm{M}-{phase}';d=read(R/(label+'.json'));assert d==ev[label]['metrics']
  assert (d['capacity'],d['query_rows'],d['fusion_layers'],d['scope'])==(128,M,2,'two_layer_probe')
  assert d['repeats']==30 and d['warmup_repeats']==3 and d['released_bytes']==0 and d['mismatched_elements']==d['differing_repeats']==0
  assert not d['npd_enabled'] and not d['io_profile_enabled'] and d['resident_copies']==0
  f=R/(label+'.f32');assert sha(f)==ev[label]['output_sha256'] and f.stat().st_size==expected*4
  v=np.fromfile(f,dtype='<f4').astype(float);out_arrays[label]=v
  ms=np.array(d['api_seconds'])*1000;assert len(ms)==30 and np.isfinite(ms).all() and min(ms)>0
  row={'case':label,'query_rows':M,'api_p50_ms':float(np.median(ms)),'api_p95_linear_ms':float(np.quantile(ms,.95)),'api_max_ms':float(max(ms)),'nonfinite_elements':int((~np.isfinite(v)).sum()),'output_sha256':sha(f)};rows.append(row)
  ports=[l for l in (R/(label+'.log')).read_text().splitlines() if l.startswith(('in[','out['))];assert len(ports)==16
  if ports0 is None:ports0=ports
  else:assert ports==ports0,label
  boundary_temps.extend(d[k]/1000 for k in ['temperature_before_mC','temperature_after_mC'])
 assert sha(R/f'm{M}-before.f32')==sha(R/f'm{M}-after.f32')
 off=0
 for name,n in meta['outputs']:
  x=out_arrays[f'm{M}-before'][off:off+n];y=out_arrays[f'm{M}-row'][off:off+n];off+=n;finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
  segments.append({'query_rows':M,'name':name,'finite':finite,'changed':int((x!=y).sum()),'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)) if finite else None,'max_abs':float(abs(x-y).max()) if finite else None})
 assert off==expected
samples=[read(R/'pre-health.json')]+[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]+[read(R/'post-health.json')]
def vm(x):return {k:int(v) for k,v in (line.split() for line in x['vmstat'].splitlines())}
res={'sampled_temperature_peak_C':max(max(x['temperatures_mC'].values()) for x in samples)/1000,'boundary_temperature_peak_C':max(boundary_temps),'min_mem_available_B':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in samples),'swapin_pages':vm(samples[-1])['pswpin']-vm(samples[0])['pswpin'],'swapout_pages':vm(samples[-1])['pswpout']-vm(samples[0])['pswpout']}
res['observed_temperature_peak_C']=max(res['sampled_temperature_peak_C'],res['boundary_temperature_peak_C'])
out={'scope':'C128 history128 L2/L3, two independent query shapes64/256, 6processes18warm180pure; same effective input only within each shape. No full model/TTFT/audio/RAG claims.','rows':rows,'segments':segments,'AA_byte_equal_each_shape':True,'port_equality_within_each_shape':True,'driver_releases_zero':6,'resources':res,'goal_achieved':False,'analysis_source_sha256':sha(Path(__file__))}
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
text='# Compact history + row-normalized down 两层筛查\n\n有效history128全部保留；query64/256是局部批次，不是完整用户输入/系统长度。每形状同输入original/row/original，6进程、18预热、180计时。仅vip_run_network，无语音/RAG，不计加载与整模head。\n\n|路径|局部API P50 ms|P95 ms（线性分位数）|非finite元素|\n|---|---:|---:|---:|\n'
for x in rows:text+=f"|{x['case']}|{x['api_p50_ms']:.4f}|{x['api_p95_linear_ms']:.4f}|{x['nonfinite_elements']}|\n"
text+='\n误差按hidden/K/V五输出分别见ANALYSIS.json。A/A完整字节同；row改变量化，不代表无损或语义准确。较晚位置须扩大历史bank，禁止丢有效历史。完整100ms目标仍未验证。\n\n资源：'+json.dumps(res,ensure_ascii=False)+'。\n'
(R/'REPORT.md').write_text(text);print(text)
