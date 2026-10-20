import unittest
from mock_executor import MockExecutor,Proposal
class Lifecycle(unittest.TestCase):
 def test_stale_and_expired_do_not_execute(self):
  x=MockExecutor(['A000','A003','A010']);sid=x.next_state();x.next_state()
  self.assertEqual(x.accept(Proposal(sid,'A010',10,.99),now=1)['status'],'stale')
  self.assertIsNone(x.accept(Proposal(x.state_id,'A010',1,.99),now=2)['action'])
 def test_hazard_overrides_model_and_uncertainty_requests_confirmation(self):
  x=MockExecutor(['A000','A003','A010']);sid=x.next_state()
  self.assertEqual(x.accept(Proposal(sid,'A010',10,.99),hazard=True,now=1)['action'],'A000')
  self.assertEqual(x.accept(Proposal(sid,'A010',10,.2),now=1)['action'],'A003')
  self.assertFalse(x.accept(Proposal(sid,'A010',10,.99),now=1)['physical_action'])
 def test_invalid_model_output_is_rejected(self):
  x=MockExecutor(['A000']);sid=x.next_state()
  self.assertEqual(x.accept(Proposal(sid,'RUN',10,.99),now=1)['status'],'invalid_action')
if __name__=='__main__':unittest.main()
