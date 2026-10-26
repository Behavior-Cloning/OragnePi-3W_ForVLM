import unittest
from preflight import evaluate,REQUIRED_BYTES
class GateTests(unittest.TestCase):
 def mem(self,n):return 'MemAvailable: %d kB\nSwapFree: 0 kB\n'%(n//1024)
 def test_active_service_blocks_even_with_memory(self):
  self.assertFalse(evaluate(self.mem(REQUIRED_BYTES+1024),'active',0)['eligible'])
 def test_low_memory_blocks(self):
  self.assertFalse(evaluate(self.mem(REQUIRED_BYTES-1024),'inactive',0)['eligible'])
 def test_nonidle_or_unknown_driver_blocks(self):
  for driver in [1,None]:
   self.assertFalse(evaluate(self.mem(REQUIRED_BYTES+1024),'inactive',driver)['eligible'])
 def test_eligible_requires_all_conditions(self):
  self.assertTrue(evaluate(self.mem(REQUIRED_BYTES+1024),'inactive',0)['eligible'])
 def test_missing_memory_does_not_pass(self):
  self.assertFalse(evaluate('','inactive',0)['eligible'])
if __name__=='__main__':unittest.main()
