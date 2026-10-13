"""A failed original must never shadow the matrix's successful retry."""
import json
from pathlib import Path
import tempfile
import unittest
from matrix_paths import cpu_result_path


class MatrixPathsTest(unittest.TestCase):
    def test_resolves_recorded_retry_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case='intern1b-h1024-n128';band='medium'
            original=root/(case+'-'+band+'-cpu');original.mkdir()
            (original/'results.json').write_text('{"status":"failed"}')
            retry=original.with_name(original.name+'-attempt2');retry.mkdir()
            (retry/'results.json').write_text('{"status":"complete"}')
            (root/'state.json').write_text(json.dumps({'cases':{case:{'bands':{band:{'cpu_path':str(retry)}}}}}))
            self.assertEqual(cpu_result_path(root,case,band),retry/'results.json')
            self.assertEqual(json.loads((original/'results.json').read_text())['status'],'failed')

    def test_missing_case_falls_back_without_guessing_newest_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'state.json').write_text('{"cases":{}}')
            self.assertEqual(cpu_result_path(root,'case','low'),root/'case-low-cpu/results.json')


if __name__=='__main__':unittest.main()
