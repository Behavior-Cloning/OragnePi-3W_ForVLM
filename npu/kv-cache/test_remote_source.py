"""Compile generated remote programs without executing board connections."""
import ast
from pathlib import Path
import unittest


class RemoteSourceTest(unittest.TestCase):
    def test_benchmark_embedded_program_is_valid_after_string_decoding(self):
        path = Path(__file__).with_name('benchmark_variants.py')
        tree = ast.parse(path.read_text())
        checked = 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'replace' and isinstance(node.func.value, ast.Constant) and
                    isinstance(node.func.value.value, str) and 'CONFIG' in node.func.value.value):
                compile(node.func.value.value.replace('CONFIG', '{}'), '<remote benchmark>', 'exec')
                checked += 1
        self.assertEqual(checked, 1)


if __name__ == '__main__':
    unittest.main()
