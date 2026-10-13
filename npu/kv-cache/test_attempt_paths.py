import json
from pathlib import Path
import tempfile
import unittest

from attempt_paths import select_attempt


class AttemptPathsTest(unittest.TestCase):
    def test_complete_is_reused_but_failed_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'cpu';base.mkdir()
            result=base/'results.json';result.write_text(json.dumps({'status':'failed'}))
            path,_,attempt=select_attempt(base,'results.json',root/'cpu.log',True)
            self.assertEqual(attempt,2);self.assertEqual(path,root/'cpu-attempt2')
            self.assertEqual(json.loads(result.read_text())['status'],'failed')
            path.mkdir();(path/'results.json').write_text(json.dumps({'status':'complete'}))
            self.assertEqual(select_attempt(base,'results.json',root/'cpu.log',True)[0],path)

    def test_log_reserves_remote_attempt_even_without_local_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'npu.log';log.write_text('remote failed before download')
            path,newlog,attempt=select_attempt(root/'npu.json',None,log,True)
            self.assertEqual((path.name,newlog.name,attempt),('npu-attempt2.json','npu-attempt2.log',2))


if __name__=='__main__':unittest.main()
