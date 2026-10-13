"""Recovery must preserve interrupted remote attempts and incomplete bands."""
import json
from pathlib import Path
import tempfile
import unittest

from run_thermal_matrix import BANDS,case_status,cpu_output


class ResumeTests(unittest.TestCase):
    def test_no_download_after_remote_stall_reserves_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old=root/'smol500-h512-n32-high-cpu-attempt3'
            self.assertFalse(old.exists())
            self.assertEqual(cpu_output(root,'smol500-h512-n32','high',str(old)),
                             root/'smol500-h512-n32-high-cpu-attempt4')

    def test_complete_result_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'case-low-cpu';old.mkdir()
            (old/'results.json').write_text(json.dumps({'status':'complete'}))
            self.assertEqual(cpu_output(root,'case','low',str(old)),old)

    def test_partial_directory_and_existing_attempt_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'case-low-cpu').mkdir()
            (root/'case-low-cpu-attempt2').mkdir()
            self.assertEqual(cpu_output(root,'case','low'),root/'case-low-cpu-attempt3')

    def test_filtered_bands_do_not_complete_case(self):
        bands={b:{'status':'complete'} for b in ('low','medium')}
        self.assertEqual(case_status(bands,{'passed':True}),'partial')
        bands['high']={'status':'blocked-storage'}
        self.assertEqual(case_status(bands,{'passed':True}),'partial')
        bands['high']['status']='complete'
        self.assertEqual(case_status(bands,{'passed':True}),'complete')

    def test_failed_gate_cannot_pass(self):
        self.assertEqual(case_status({b:{'status':'complete'} for b in BANDS},
                                     {'passed':False}),'quality-failed')


if __name__=='__main__':unittest.main()
