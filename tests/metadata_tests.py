import os
import pickle
import mock
import pandas as pd
import unittest
#import interactive_data_tree as idt
#from interactive_data_tree.interactive_data_tree import interactive_data_tree as idt
import tempfile
import shutil
import sys
import collections
from .context import interactive_data_tree as idt

EXEC_PYTHON3 = sys.version_info > (3, 0)
# Patching the builtin directly doesn't seem to always work in Python 2
builtins_input_mock_str = 'interactive_data_tree.interactive_data_tree.prompt_input'

class MetadataTests(unittest.TestCase):
    @staticmethod
    def get_temp_dir():
        return tempfile.mkdtemp()

    def setUp(self):
        self.repo_root_path = MetadataTests.get_temp_dir()
        self.repo_root_path_b = MetadataTests.get_temp_dir()
        self.rt = idt.RepoTree(self.repo_root_path)

        self.md_dir = MetadataTests.get_temp_dir()
        self.md_path = self.md_dir + '/' + 'unittest.md'
        self.md_lock = self.md_dir + '/' + 'unittest.lock'
        self.md_path_b = self.md_dir + '/' + 'unittest_b.md'

    def tearDown(self):
        shutil.rmtree(self.repo_root_path)
        shutil.rmtree(self.repo_root_path_b)

        shutil.rmtree(self.md_path, ignore_errors=True)
        shutil.rmtree(self.md_path_b, ignore_errors=True)

    def test_init(self):
        md = idt.Metadata(path=self.md_path)
        md = idt.Metadata(path=self.md_path, lock_path=self.md_lock)
        md = idt.Metadata(path=self.md_path, lock_path=self.md_lock,
                          required_fields=['foo', 'bar'])
        md = idt.Metadata(path=self.md_path, lock_path=self.md_lock,
                          required_fields=['foo', 'bar'], resolve_tree=self.rt)

    def test_exists_touch_delete(self):
        md = idt.Metadata(path=self.md_path, lock_path=self.md_lock,
                          required_fields=['foo', 'bar'], resolve_tree=self.rt)
        self.assertFalse(md.exists(),
                         msg='Exists should return True immediately after ctor')

        md.touch()
        self.assertTrue(md.exists(),
                        msg='Metadata should exist after touch()')
        md.remove()
        self.assertFalse(md.exists(),
                        msg='Metadata should not exist after remove()')


    def test_basic_write_read(self):
        md = idt.Metadata(path=self.md_path)
        kv = dict(foo=5, bar='thing')
        idt_md = dict(#tree_md=dict(),
                      #si_md=dict(),
                      user_md=kv)
        md.write_metadata(**idt_md)
        last_md = md.read_metadata(most_recent=True, lock=True)



# TODO:
# - Handle wrong types and check types within reason (e.g. strings!)
# - Basic search functionality off of tree
#   - Maybe use Wikipedia vocab to do n-gram similatiry between search and doc
#   - Do similarity between code cells?
#       - Writing something..."one of us has done this"...code simalrity
# - Test with Python2
if __name__ == '__main__':
    unittest.main()
