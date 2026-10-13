"""Protocol faults must fail closed without requiring a board or SDK."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from resident_worker import ResidentWorker


FAKE = r'''
import json,os,struct,sys,time
events=os.fdopen(int(os.environ['OPI_SESSION_FD']),'w',buffering=1)
def emit(event):
    # Deliberately split each JSON line across multiple writes.
    encoded=json.dumps(event)+'\n';cut=len(encoded)//2
    events.write(encoded[:cut]);events.flush();events.write(encoded[cut:]);events.flush()
print('SDK diagnostics on stdout',flush=True)
emit({'event':'ready','capacity':16})
while True:
    header=sys.stdin.buffer.read(32)
    if not header:break
    magic,i,n,limit,nf,reset,diagnostic,reserved=struct.unpack('<8I',header)
    assert magic==0x4f504931 and reserved==0
    raw=sys.stdin.buffer.read(4*(n+nf));assert len(raw)==4*(n+nf)
    tokens=struct.unpack('<'+'i'*(n+nf),raw)
    if tokens[0]==9:
        time.sleep(30)
    elif tokens[0]==8:
        emit({'event':'token','id':i+1,'token':7});continue
    for t in range(limit):emit({'event':'token','id':i,'token':7})
    emit({'event':'done','id':i,'generated_tokens':limit,'prompt_tokens':n,
          'reused_tokens':0,'new_tokens':n,'reset':reset,'diagnostic':diagnostic})
emit({'event':'closed','driver_bytes':0})
'''


class ResidentProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = ResidentWorker([sys.executable, '-u', '-c', FAKE], self.root,
                                     self.root / 'worker.log')

    def tearDown(self):
        self.worker.abort()
        self.temporary.cleanup()

    def test_split_events_and_multiple_requests(self):
        first = list(self.worker.generate([1, 2, 3], 2, feedback=[4, 5], diagnostic=True))
        self.assertEqual(first[-1]['tokens'], [7, 7])
        self.assertEqual(first[-1]['id'], 1)
        self.assertTrue(first[-1]['diagnostic'])
        second = list(self.worker.generate([1, 2], 1, reset=True))
        self.assertEqual(second[-1]['id'], 2)
        self.assertTrue(second[-1]['reset'])
        self.assertEqual(self.worker.close()['driver_bytes'], 0)
        self.assertIn('SDK diagnostics', (self.root / 'worker.log').read_text())

    def test_overflow_does_not_send_request(self):
        with self.assertRaises(ValueError):
            list(self.worker.generate([1] * 15, 3))
        done = list(self.worker.generate([1], 1))[-1]
        self.assertEqual(done['id'], 1)
        self.worker.close()

    def test_mismatched_id_terminates_worker(self):
        with self.assertRaisesRegex(RuntimeError, 'ID mismatch'):
            list(self.worker.generate([8], 1))
        self.assertIsNotNone(self.worker.process.poll())

    def test_stalled_worker_deadline(self):
        with self.assertRaises(TimeoutError):
            list(self.worker.generate([9], 1, timeout=.1))
        self.assertIsNotNone(self.worker.process.poll())


if __name__ == '__main__':
    unittest.main()
