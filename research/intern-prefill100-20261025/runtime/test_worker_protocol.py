"""Local fake-worker protocol regression. No NPU/board/performance claim."""
import json,sys,os,struct,tempfile,unittest
from pathlib import Path
from resident_worker import ResidentWorker

def fake():
 out=os.fdopen(int(os.environ['OPI_SESSION_FD']),'w',buffering=1)
 def emit(d):out.write(json.dumps(d)+'\n')
 emit({'event':'ready','capacity':512});emit({'event':'runtime_contract','protocol':2})
 while True:
  h=sys.stdin.buffer.read(32)
  if not h:break
  assert len(h)==32
  magic,request,n,limit,nf,reset,flags,prefix=struct.unpack('<8I',h)
  assert magic==0x4f504932 and prefix in (64,128) and n>=prefix
  data=sys.stdin.buffer.read(4*(n+nf));assert len(data)==4*(n+nf)
  reuse=prefix if n>prefix else 0
  if os.environ.get('FAKE_BAD_REUSE'):reuse+=1
  for i in range(limit):emit({'event':'token','id':request,'token':1,'elapsed':i*0.001})
  emit({'event':'done','id':request,'generated_tokens':limit,'prompt_tokens':n,'prefix_tokens':prefix,'reused_tokens':reuse,'new_tokens':n-reuse})
 emit({'event':'closed','driver_bytes':0})

class Tests(unittest.TestCase):
 def worker(self,d,bad=False):
  cmd=[sys.executable,str(Path(__file__).resolve()),'fake']
  if bad:cmd=['env','FAKE_BAD_REUSE=1',*cmd]
  return ResidentWorker(cmd,d,str(Path(d)/'worker.log'),timeout=5)
 def test_roundtrip_prefix_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   w=self.worker(d)
   try:
    for prefix in [64,128,64]:
     a=list(w.generate([1]*prefix,max_new=1,reset=True,prefix_tokens=prefix,timeout=5))[-1]
     self.assertEqual(a['reused_tokens'],0)
     for body in [64,128,192,256]:
      a=list(w.generate([1]*(prefix+body+8),max_new=32,feedback=[1]*32,prefix_tokens=prefix,timeout=5))[-1]
      self.assertEqual((a['reused_tokens'],a['new_tokens']),(prefix,body+8))
    self.assertEqual(w.close()['driver_bytes'],0)
   finally:w.abort()
 def test_missing_prefix_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   w=self.worker(d)
   try:
    with self.assertRaises(ValueError):list(w.generate([1]*100,max_new=1))
   finally:w.abort()
 def test_wrong_reuse_rejected_and_worker_killed(self):
  with tempfile.TemporaryDirectory() as d:
   w=self.worker(d,True)
   with self.assertRaises(RuntimeError):list(w.generate([1]*136,max_new=1,prefix_tokens=64,timeout=5))
   self.assertIsNotNone(w.process.poll())
if __name__=='__main__':
 if sys.argv[-1]=='fake':fake()
 else:unittest.main()
