"""Mock-only decision lifecycle; no device or motor interfaces."""
from dataclasses import dataclass
import time
@dataclass(frozen=True)
class Proposal:
 state_id:int
 action:str
 deadline:float
 confidence:float
class MockExecutor:
 def __init__(self,valid_actions,confidence_threshold=.8):
  self.valid_actions=set(valid_actions);self.threshold=confidence_threshold;self.state_id=0;self.audit=[]
 def next_state(self):self.state_id+=1;return self.state_id
 def accept(self,p,hazard=False,now=None):
  now=time.monotonic() if now is None else now
  if p.state_id!=self.state_id:status,action='stale',None
  elif now>p.deadline:status,action='expired',None
  elif hazard:status,action='safety_override','A000'
  elif p.action not in self.valid_actions:status,action='invalid_action',None
  elif not 0<=p.confidence<=1:status,action='invalid_confidence',None
  elif p.confidence<self.threshold:status,action='needs_confirmation','A003'
  else:status,action='mock_ack',p.action
  result=dict(status=status,action=action,physical_action=False,state_id=p.state_id,at=now);self.audit.append(result);return result
