"""Download official release assets and record provenance; keep weights outside Git."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
root=Path('/mnt/f/Files/OPi-Lab/audio-models-20261018');root.mkdir(exist_ok=True)
base='https://github.com/k2-fsa/sherpa-onnx/releases/download/'
assets=[
 ('vad','asr-models/silero_vad.onnx'),
 ('denoise','speech-enhancement-models/gtcrn_simple.onnx'),
 ('denoise_sample','speech-enhancement-models/speech_with_noise.wav'),
 ('speaker','speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'),
 ('asr','asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2'),
]
def download(item):
 category,path=item;url=base+path;p=root/path.split('/')[-1];tmp=p.with_suffix(p.suffix+'.partial')
 if not p.exists():
  with urllib.request.urlopen(url,timeout=60) as r,tmp.open('wb') as f:
   while data:=r.read(1024*1024):f.write(data)
  tmp.replace(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 result={'category':category,'url':url,'filename':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(download,assets))
(root/'assets.json').write_text(json.dumps(result,indent=2)+'\n')
repo=Path(__file__).resolve().parents[2]
(repo/'results/audio-models-20261018/assets.json').write_text(json.dumps(result,indent=2)+'\n')
