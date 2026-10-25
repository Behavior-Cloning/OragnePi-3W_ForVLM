import copy,importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('benchmark',P/'board_benchmark.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FixtureMetadata(unittest.TestCase):
 def setUp(self):self.doc=json.loads((P/'aligned-inputs.json').read_text())
 def test_original_quality_metadata_attached_without_changing_inputs(self):
  old=copy.deepcopy(self.doc);m.attach_system_metadata(self.doc)
  for before,after in zip(old['performance']+old['quality'],self.doc['performance']+self.doc['quality']):
   self.assertEqual(before['tokens'],after['tokens']);self.assertEqual(before['tokens_sha256'],after['tokens_sha256'])
   self.assertEqual(after['static_tokens'],after['system_body_tokens']+after['system_template_tokens'])
   self.assertEqual(after['system_template_tokens'],5)
 def test_wrong_prefix_rejected(self):
  self.doc['quality'][0]['prefix_tokens'][0]+=1
  with self.assertRaises(AssertionError):m.attach_system_metadata(self.doc)
 def test_wrong_body_count_rejected(self):
  self.doc['quality'][0]['system_body_tokens']=999
  with self.assertRaises(AssertionError):m.attach_system_metadata(self.doc)
 def test_unknown_prefix_rejected(self):
  self.doc['quality'][0]['prefix_sha256']='0'*64
  with self.assertRaises(KeyError):m.attach_system_metadata(self.doc)
if __name__=='__main__':unittest.main()
