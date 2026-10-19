"""Exercise publication policy without board-only model dependencies."""
import ast,unittest
from pathlib import Path
src=ast.parse(Path(__file__).with_name('two_stage.py').read_text())
ns={};exec(compile(ast.Module(body=[n for n in src.body if isinstance(n,ast.ClassDef) and n.name=='Gate'],type_ignores=[]),'gate','exec'),ns);Gate=ns['Gate']
class PublicationTests(unittest.TestCase):
 def test_tentative_never_public(self):
  g=Gate();g.advance(1);self.assertFalse(g.accepts(1,[1,2]))
 def test_matching_confirmed_prompt_allowed(self):
  g=Gate();g.advance(1);self.assertTrue(g.confirm(1,[1,2]));self.assertTrue(g.accepts(1,[1,2]))
 def test_resume_revokes_confirmation_and_late_results(self):
  g=Gate();g.advance(1);g.confirm(1,[1,2]);g.advance(2);self.assertFalse(g.accepts(1,[1,2]));self.assertFalse(g.confirm(1,[1,2]));self.assertFalse(g.accepts(2,[1,2]))
 def test_changed_transcript_cannot_reuse_candidate(self):
  g=Gate();g.advance(1);g.confirm(1,[1,3]);self.assertFalse(g.accepts(1,[1,2]))
 def test_old_event_cannot_rewind_revision(self):
  g=Gate();g.advance(2);g.advance(1);self.assertEqual(g.revision,2)
if __name__=='__main__':unittest.main()
