"""Freeze public WAV metadata and add deterministic unclipped synthetic noise."""
import array,hashlib,json,math,random,tarfile,wave
from pathlib import Path
root=Path('/mnt/f/Files/OPi-Lab/asr-small-models-20261018');out=root/'corpus';out.mkdir(exist_ok=True)
with tarfile.open(root/'corpus-clean.tar') as t:
 for m in t.getmembers():assert Path(m.name).name==m.name and not m.issym() and not m.islnk()
 t.extractall(out)
d=json.loads((out/'manifest.json').read_text());original=list(d['samples']);rng=random.Random(20261018)
for idx in [0,8,9,16]:
 r=original[idx]
 with wave.open(str(out/r['file'])) as w:x=[v/32768 for v in array.array('h',w.readframes(w.getnframes()))]
 noise=[rng.gauss(0,1) for _ in x];rms=math.sqrt(sum(z*z for z in x)/len(x));nrms=math.sqrt(sum(z*z for z in noise)/len(noise))
 for snr in [10,0]:
  n=[z*rms/nrms*10**(-snr/20) for z in noise];y=[a+b for a,b in zip(x,n)];scale=max(1,max(map(abs,y))/.99);name=r['id']+f'-white-{snr}db.wav'
  with wave.open(str(out/name),'w') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(array.array('h',[round(z/scale*32767) for z in y]).tobytes())
  d['samples'].append(dict(r,id=Path(name).stem,file=name,condition=f'white-{snr}db',snr_db=snr,common_scale=scale,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
with wave.open(str(out/'silence.wav'),'w') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(96000))
d['samples'].append({'id':'silence','file':'silence.wav','text':'','duration_seconds':3,'rate':16000,'condition':'silence','sha256':hashlib.sha256((out/'silence.wav').read_bytes()).hexdigest()})
d['acceptance_before_results']={'screening_only':True,'qwen_trigger':'If none of stream/offline/para has clean CER <=10% and preserves all selected negation/date key strings, deploy and test Qwen0.6B. Small test cannot establish production accuracy.','critical_strings':{original[i]['id']:words for i,words in [(0,['不可']),(1,['不适']),(2,['没有']),(3,['不少']),(4,['不能','无关']),(5,['没有']),(6,['两次','第三次']),(8,['二零二二年']),(9,['五月七日']),(14,['李嘉诚']),(16,['王思聪','朱圣祎'])]}}
(out/'manifest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(len(d['samples']),sum(r['duration_seconds'] for r in d['samples']))
