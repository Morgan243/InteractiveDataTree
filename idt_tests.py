import os
import pickle
import mock
import pandas as pd
import unittest
import interactive_data_tree as idt
import tempfile
import shutil
import sys

EXEC_PYTHON3 = sys.version_info > (3, 0)
# Patching the builtin directly doesn't seem to always work in Python 2
builtins_input_mock_str = 'interactive_data_tree.prompt_input'

class InteractiveDataRepo(unittest.TestCase):
    @staticmethod
    def get_temp_dir():
        return tempfile.mkdtemp()

    def setUp(self):
        self.repo_root_path = InteractiveDataRepo.get_temp_dir()

    def tearDown(self):
        shutil.rmtree(self.repo_root_path)

    def test_repo_creation(self):
        rep_tree = idt.RepoTree(repo_root=self.repo_root_path)
        self.assertEqual(rep_tree.idr_prop['repo_root'], self.repo_root_path + '.repo')
        rep_tree.mkrepo(name='lvl1a')

        self.assertTrue(hasattr(rep_tree, 'lvl1a'))
        repos, objs = rep_tree.list()
        self.assertEqual(repos, ['lvl1a'])

        with self.assertRaises(ValueError):
            rep_tree.mkrepo(name='lvl1a', err_on_exists=True)

    def test_obj_save_overwrite_delete(self):
        rep_tree = idt.RepoTree(repo_root=self.repo_root_path)
        #rep_tree.mkrepo(name='lvl1a')
        t_obj = 'some string'
        rep_tree.save(t_obj, 'test_string')

        self.assertTrue(hasattr(rep_tree, 'test_string'))
        repos, objs = rep_tree.list()
        self.assertEqual(['LOG', 'test_string'], objs)

        self.assertEqual(t_obj, rep_tree.load('test_string'))
        self.assertEqual(t_obj, rep_tree.test_string.load(storage_type='pickle'))
        self.assertEqual(t_obj, rep_tree.test_string.load(storage_type=None))
        self.assertEqual(t_obj, rep_tree.test_string())

        # Test aborting the operation ('n') - check it didn't do through after
        with mock.patch(builtins_input_mock_str, return_value='n'):
            n_t_obj = 'another string'
            rep_tree.save(n_t_obj, 'test_string')

        self.assertEqual(t_obj, rep_tree.load('test_string'))
        self.assertEqual(t_obj, rep_tree.test_string.load())
        self.assertEqual(t_obj, rep_tree.test_string())

        with mock.patch(builtins_input_mock_str, return_value='y'):
            n_t_obj = 'another string'
            rep_tree.save(n_t_obj, 'test_string')

        self.assertEqual(n_t_obj, rep_tree.load('test_string'))
        self.assertEqual(n_t_obj, rep_tree.test_string.load())
        self.assertEqual(n_t_obj, rep_tree.test_string())

        # Add another obj to test that a deletion hits only the target object
        t_obj = 'some string2'
        rep_tree.save(t_obj, 'test_string_num_2')
        rep_tree.test_string.delete(author='unittests')
        self.assertTrue(not hasattr(rep_tree, 'test_string'))
        self.assertTrue(hasattr(rep_tree, 'test_string_num_2'))
        rep_tree.test_string_num_2.load()

        rep_tree.delete(author='unittests', name='test_string_num_2')
        self.assertTrue(not hasattr(rep_tree, 'test_string_num_2'))

    def test_load_exceptions(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        with self.assertRaises(ValueError):
            rt.load('something not there')

    def test_listing(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('lvl1')
        rt.lvl1.mkrepo('lvl2')
        t_str = 'foo bar'
        rt.lvl1.save(t_str, name='just_a_string')

        repos = rt.lvl1.list(list_objs=False)
        self.assertEqual(['lvl2'], repos)

        objs = rt.lvl1.list(list_repos=False)
        self.assertEqual(['just_a_string'], objs)

        with self.assertRaises(ValueError):
            rt.lvl1.list(list_objs=False, list_repos=False)

    def test_pandas_storage(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1.save(df, 'test_df', storage_type='hdf')
        ld_df = lvl1.test_df.load()
        pd.util.testing.assert_frame_equal(df, ld_df)

        lvl1.save(df.a, 'test_series', storage_type='hdf')
        ld_s = lvl1.test_series.load()
        pd.util.testing.assert_series_equal(df.a, ld_s)

    def test_pandas_sample(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1 = rt.mkrepo('lvl1')
        lvl1.save(df, name='test_df')
        s_df = lvl1.test_df.hdf.sample(n=5)
        pd.util.testing.assert_frame_equal(df.head(5), s_df)

    def test_multiplex_storage(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1.save(df, 'test_df', storage_type='hdf')
        lvl1.save('not a df', 'test_df', storage_type='pickle')

        ld_df = lvl1.test_df.load()
        pd.util.testing.assert_frame_equal(df, ld_df)
        pd.util.testing.assert_frame_equal(df, lvl1.test_df.hdf.load())
        self.assertEqual('not a df', lvl1.test_df.pickle.load())

        lvl1.save(df.a, 'test_series', storage_type='hdf')
        ld_s = lvl1.test_series.load()
        pd.util.testing.assert_series_equal(df.a, ld_s)

        lvl1.test_df.delete(author='unittests', storage_type='hdf')
        self.assertEqual('not a df', lvl1.test_df.load())

    def test_attribute_axis_race(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)

        with self.assertRaises(AttributeError):
            rt.test_str.load()

        t_str = 'foobar'
        tmp_path = os.path.join(rt.idr_prop['repo_root'], 'test_str.pkl')
        pickle.dump(t_str, open(tmp_path, 'wb'))

        self.assertTrue(hasattr(rt, 'test_str'))
        self.assertEqual(t_str, rt.test_str.load())

        tmp_path = os.path.join(rt.idr_prop['repo_root'], 'test_repo.repo')
        os.mkdir(tmp_path)
        self.assertTrue(hasattr(rt, 'test_repo'))

    def test_parent_repo_listing(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl4 = rt.mkrepo('lvl1').mkrepo('lvl2').mkrepo('lvl3').mkrepo('lvl4')
        pr = lvl4.get_parent_repo_names()
        expected_pr = ['lvl1', 'lvl2', 'lvl3']
        self.assertEqual(expected_pr, pr[1:])

    def test_summary(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('lvl1')
        rt.summary()

    def test_default_init(self):
        rt = idt.RepoTree(repo_root=None)

    def test_metadata(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1.save(df, 'test_df', storage_type='hdf', comments='foobar', author='tester')
        ld_df = lvl1.test_df.load()

        self.assertEqual('foobar', lvl1.test_df.read_metadata()[-1]['comments'])
        self.assertEqual('tester', lvl1.test_df.read_metadata(storage_type='hdf')[-1]['author'])

    def test_getitem(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        self.assertEqual(lvl1, rt['lvl1'])
        lvl3 = lvl1.mkrepo('lvl2').mkrepo('lvl3')
        self.assertEqual(lvl3, rt['lvl1']['lvl2']['lvl3'])
        lvl3.save('some string data', name='foobar')
        self.assertEqual(lvl3.foobar, lvl3['foobar'])
        self.assertEqual(lvl3.foobar, rt['lvl1']['lvl2']['lvl3']['foobar'])

        with self.assertRaises(KeyError):
            t = lvl1['not_there']

    def test_html_repr(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1.save(df, 'test_df',
                  storage_type='hdf')
        lvl1.save('not a df', 'test_str',
                  storage_type='pickle')

        lvl1._repr_html_()
        lvl1.test_df._repr_html_()
        lvl1.test_str._repr_html_()

    def test_name_collisions(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        rt.save('str object', name='lvl1')

        # Repo takes precedent
        self.assertEqual(lvl1, rt.lvl1)
        self.assertEqual(lvl1, rt['lvl1'])

        # Objects can be accessed through the load interface
        self.assertEqual(rt.load('lvl1'), 'str object')

    def test_invalid_identifier(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        with self.assertRaises(ValueError):
            lvl1 = rt.mkrepo('lvl1 invalid')

        with self.assertRaises(ValueError):
            rt.save('some obj', 'invalid id')

    def test_queryable_extractions(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.save('str object', name='some_data',
                comments='something to search for')

        terms = rt.some_data.pickle.get_terms()
        self.assertIn('str', terms)
        self.assertIn('something to search for', terms)

        vec_dict = rt.some_data.pickle.get_vector_representation()

        self.assertIsInstance(vec_dict, dict)

    def test_ipython_features(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.save('str object', name='some_data',
                comments='something to search for')

        lvl1 = rt.mkrepo('lvl1')
        rt.save('some string data', name='foobar')

        comps = rt._ipython_key_completions_()
        self.assertIn('lvl1', comps)
        self.assertIn('foobar', comps)
        self.assertIn('some_data', comps)
        self.assertEqual(len(comps), 4)

    def test_unsupported_object_exceptions(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        with self.assertRaises(ValueError):
            rt.save('not a dataframe',
                    name='foobar', storage_type='hdf')

        if EXEC_PYTHON3:
            with self.assertRaises(AttributeError):
                rt.save(lambda x: x*5,
                        name='foobar_lambda', storage_type='pickle')
        else:
            with self.assertRaises(pickle.PicklingError):
                rt.save(lambda x: x*5,
                        name='foobar_lambda', storage_type='pickle')

# TODO:
# - Handle wrong types and check types within reason (e.g. strings!)
# - Basic search functionality off of tree
#   - Maybe use Wikipedia vocab to do n-gram similatiry between query and doc
#   - Do similarity between code cells?
#       - Writing something..."one of us has done this"...code simalrity
# - Test with Python2
if __name__ == '__main__':
    unittest.main()
