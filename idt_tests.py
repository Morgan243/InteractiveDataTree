import os
import pickle
import mock
import pandas as pd
import unittest
import interactive_data_tree as idt
import tempfile
import shutil
import sys
import collections

EXEC_PYTHON3 = sys.version_info > (3, 0)
# Patching the builtin directly doesn't seem to always work in Python 2
builtins_input_mock_str = 'interactive_data_tree.prompt_input'

class InteractiveDataRepo(unittest.TestCase):
    @staticmethod
    def get_temp_dir():
        return tempfile.mkdtemp()

    def setUp(self):
        self.repo_root_path = InteractiveDataRepo.get_temp_dir()
        self.repo_root_path_b = InteractiveDataRepo.get_temp_dir()

    def tearDown(self):
        shutil.rmtree(self.repo_root_path)
        shutil.rmtree(self.repo_root_path_b)

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
        self.assertEqual(['INDEX', 'LOG', 'test_string'], objs)

        self.assertEqual(t_obj, rep_tree.load('test_string'))
        self.assertEqual(t_obj, rep_tree.test_string.load(storage_type='pickle'))
        self.assertEqual(t_obj, rep_tree.test_string.pickle())
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

    def test_save_exceptions(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        with self.assertRaises(ValueError):
            rt.save('something not there', name='foobar',
                    storage_type='no exist')

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

    def test_reinit(self):
        # Test that reopening a repo populates correctly
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a').save('str object', name='some_data',
                comments='something to search for')
        rt.subrepo_a.save('foobar object', name='other_data',
                comments='12nm 1n2d 121 23j')
        rt.mkrepo('subrepo_b').save('foobar thing', name='more_data',
                comments=',mncxzlaj aois mas na')

        rt2 = idt.RepoTree(repo_root=self.repo_root_path)
        self.assertTrue('other_data' in dir(rt2.subrepo_a))
        self.assertTrue('subrepo_b' in dir(rt2))

    def test_metadata(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1.save(df, 'test_df', storage_type='hdf', comments='foobar', author='tester')
        ld_df = lvl1.test_df.load()

        self.assertEqual('foobar', lvl1.test_df.read_metadata()['comments'])
        self.assertEqual('tester', lvl1.test_df.read_metadata(storage_type='hdf')['author'])

    def test_str_reference(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        lvl2 = lvl1.mkrepo('lvl2')
        lvl2.save('lvl 2 string', name='barfoo')

        lvl3 = lvl2.mkrepo('lvl3')
        lvl3.save('some string data', name='foobar')

        lvl1.save('some more data', name='barmoo',
                  other=lvl2.barfoo,
                  references=lvl3.foobar)

        ref_str = lvl3.foobar.reference()
        self.assertEqual(lvl3.foobar.pickle, rt.from_reference(ref_str))

        # reference interface is overloaded to handle either string or StorageInterface objects
        # If passed a terminating node, make sure the node is just returned
        self.assertEqual(lvl3.foobar.pickle, rt.from_reference(lvl3.foobar.pickle))

        # make sure references are handled on instantiation of new repo
        rt_a = idt.RepoTree(repo_root=self.repo_root_path)

        rt_b = idt.RepoTree(repo_root=self.repo_root_path_b)
        rt_b.save('some object data', name='barfoo')
        with self.assertRaises(ValueError):
            rt.from_reference(rt_b.barfoo.pickle)

        fake_reference = 'rootishrepo-subrepo-subsubrepo-data-pickle'
        with self.assertRaises(ValueError):
            rt.from_reference(fake_reference)

    def test_getitem(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        self.assertEqual(lvl1, rt['lvl1'])
        lvl3 = lvl1.mkrepo('lvl2').mkrepo('lvl3')
        self.assertEqual(lvl3, rt['lvl1']['lvl2']['lvl3'])
        lvl3.save('some string data', name='foobar')
        self.assertEqual(lvl3.foobar, lvl3['foobar'])
        self.assertEqual(lvl3.foobar, rt['lvl1']['lvl2']['lvl3']['foobar'])
        self.assertEqual(lvl3.foobar.pickle, rt['lvl1']['lvl2']['lvl3']['foobar']['pickle'])

        with self.assertRaises(KeyError):
            t = lvl1['not_there']

        with self.assertRaises(KeyError):
            t = rt['lvl1']['lvl2']['lvl3']['foobar']['hdf']

    def test_setitem(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        with self.assertRaises(NotImplementedError):
            rt['foobar_str'] = 'some string data'

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
                comments='SoMeThInG to search for')

        terms = rt.some_data.pickle.get_terms()
        self.assertIn('str', terms)
        self.assertIn('something', terms)
        self.assertIn('search', terms)

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
        self.assertEqual(len(comps), 5)

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

    def test_query(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a').save('str object', name='some_data',
                comments='something to search for')
        rt.subrepo_a.save('foobar object', name='other_data',
                comments='12nm 1n2d 121 23j')
        rt.mkrepo('subrepo_b').save('foobar thing', name='more_data',
                comments=',mncxzlaj aois mas na')

        res = rt.query('something to search for')
        k = idt.URI_SPEC + rt.name + '/subrepo_a/some_data/pickle'
        self.assertTrue(res[k] == max(res.values()))

        res = rt.query('12nm 121 23j 1n2d')
        k = idt.URI_SPEC + rt.name + '/subrepo_a/other_data/pickle'
        self.assertTrue(res[k] == max(res.values()))

        ref = rt.subrepo_a.some_data.reference()
        rt.subrepo_a.some_data.delete(author='unittests')
        self.assertTrue(ref not in rt.INDEX())

    def test_model_storage(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a')
        rt.subrepo_a.save('str object', name='some_data',
                           comments='something to search for')

        from sklearn.linear_model import LogisticRegression
        import numpy as np

        X = np.random.rand(100, 10)
        y = np.random.rand(100).round()

        X_cv = np.random.rand(100, 10)
        y_cv = np.random.rand(100).round()

        m = LogisticRegression().fit(X, y)
        features = ['a', 'b', 'c']
        target = 'z'

        rt.save(m, name='logit_model',
                input_data=rt.subrepo_a.some_data,
                features=features,
                target=target,
                storage_type='model')

        rt.logit_model.model.predict(X_cv)
        rt.logit_model.model.predict_proba(X_cv)
        self.assertEqual(rt.logit_model.features, features)
        self.assertEqual(rt.logit_model.target, target)

    def test_sql_storage(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a').save('str object', name='some_data',
                                    comments='something to search for')

        sql_obj = idt.SQL(select_statement='id, name, age',
                          from_statement='db.table',
                          where_statement='')

        rt.subrepo_a.save(sql_obj,
                          name='query_name_and_age')

        with self.assertRaises(ValueError):
            rt.subrepo_a.query_name_and_age.query(dict())

        query = lambda x: x

        CXN = collections.namedtuple('CXN', 'query')
        cxn = CXN(query=query)
        rt.subrepo_a.query_name_and_age.query(cxn)

    def test_interface_registration(self):
        # TODO: This test leaves artifacts that may impact other tests
        # as it registers a new interface. Consider wiping before after tests
        class TestInterface(idt.StorageInterface):
            storage_name = 'test'
            required_metadata = ['random_req']
            extension = 'tst'

        idt.register_storage_interface(TestInterface, name='test',
                                       priority=None, types=None)

# TODO:
# - Handle wrong types and check types within reason (e.g. strings!)
# - Basic search functionality off of tree
#   - Maybe use Wikipedia vocab to do n-gram similatiry between query and doc
#   - Do similarity between code cells?
#       - Writing something..."one of us has done this"...code simalrity
# - Test with Python2
if __name__ == '__main__':
    unittest.main()
