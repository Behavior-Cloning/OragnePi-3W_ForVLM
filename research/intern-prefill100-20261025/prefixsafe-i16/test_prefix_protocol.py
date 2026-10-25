import io,struct,types,unittest
from resident_worker_v1 import ResidentWorker
class PrefixProtocol(unittest.TestCase):
 def worker(self,n):
  w=object.__new__(ResidentWorker);w.capacity=1280;w.last_request=0;w.process=types.SimpleNamespace(stdin=io.BytesIO())
  events=iter([{'id':1,'event':'token','token':42},{'id':1,'event':'done','generated_tokens':1,'prompt_tokens':n}]);w._event=lambda deadline:next(events)
  w.abort=lambda:None
  return w
 def test_explicit_whole_prefix(self):
  w=self.worker(64);list(w.generate([1]*64,1,reset=True,original_prefix_tokens=64));header=struct.unpack('<8I',w.process.stdin.getvalue()[:32]);self.assertEqual(header,(0x4F504931,1,64,1,0,1,0,64))
 def test_legacy_zero_unchanged(self):
  w=self.worker(136);list(w.generate([1]*136,1));self.assertEqual(struct.unpack('<8I',w.process.stdin.getvalue()[:32])[-1],0)
 def test_reject_boundary_change_on_continuation(self):
  w=self.worker(64)
  with self.assertRaises(ValueError):list(w.generate([1]*64,1,original_prefix_tokens=64))
  self.assertEqual(w.process.stdin.getvalue(),b'')
 def test_reject_partial_or_noninteger_prefix(self):
  for length in [32,-1,64.0,True]:
   w=self.worker(64)
   with self.assertRaises(ValueError):list(w.generate([1]*64,1,reset=True,original_prefix_tokens=length))
   self.assertEqual(w.process.stdin.getvalue(),b'')
if __name__=='__main__':unittest.main()
