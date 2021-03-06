import unittest
import tempfile
import shutil
import sys
import pandas as pd
import collections
from .context import interactive_data_tree as idt

EXEC_PYTHON3 = sys.version_info > (3, 0)
# Patching the builtin directly doesn't seem to always work in Python 2
builtins_input_mock_str = 'interactive_data_tree.interactive_data_tree.prompt_input'

class StorageInterfaceTests(unittest.TestCase):
    @staticmethod
    def get_temp_dir():
        return tempfile.mkdtemp()

    def setUp(self):
        self.repo_root_path = StorageInterfaceTests.get_temp_dir()
        self.repo_root_path_b = StorageInterfaceTests.get_temp_dir()
        self.rt = idt.RepoTree(self.repo_root_path)

        self.md_dir = StorageInterfaceTests.get_temp_dir()
        self.md_path = self.md_dir + '/' + 'unittest.md'
        self.md_lock = self.md_dir + '/' + 'unittest.lock'
        self.md_path_b = self.md_dir + '/' + 'unittest_b.md'

    def tearDown(self):
        shutil.rmtree(self.repo_root_path)
        shutil.rmtree(self.repo_root_path_b)

        shutil.rmtree(self.md_path, ignore_errors=True)
        shutil.rmtree(self.md_path_b, ignore_errors=True)

    def test_init(self):
        with self.assertRaises(ValueError):
            si = idt.StorageInterface('foo')

    def test_multi_metadata(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        n = 10
        for v in range(n):
            rt.save(v, name='multiwrite', comments='this is v=%d' % v,
                    auto_overwrite=True)

        mds = rt['multiwrite'].read_metadata(most_recent=False)
        # TODO: Metadata has an empty first entry, adding one to length
        self.assertEqual(len(mds), n+1)

    def test_bad_storage_registration(self):
        class BadSI:
            pass
        with self.assertRaises(ValueError):
            idt.register_storage_interface(BadSI, 'BadSI')

        with self.assertRaises(ValueError):
            idt.register_storage_interface(idt.HDFStorageInterface, 'hdf')

        class HDFExtra(idt.HDFStorageInterface):
            extension = 'hdf_extra'
            pass
        #with self.assertRaises(ValueError):
        idt.register_storage_interface(HDFExtra,
                                           'hdf_extra',
                                       types=pd.DataFrame)

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

    def test_pandas_group_hdf(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        lvl1 = rt.mkrepo('lvl1')
        df_grps = {1999 : pd.DataFrame(dict(a=range(100), b=range(100, 200))),
         2000 : pd.DataFrame(dict(a=range(100), b=range(100, 200))),
         2001 : pd.DataFrame(dict(a=range(100), b=range(100, 200))),
         }
        lvl1.save(df_grps, 'test_gdf', storage_type='ghdf',
                  extra_md_stuff='foobar')
        pd.util.testing.assert_frame_equal(df_grps[1999],
                                           lvl1.test_gdf[1999])
        pd.util.testing.assert_frame_equal(df_grps[2000],
                                           lvl1.test_gdf[2000])
        concat_df = pd.concat([df_grps[1999], df_grps[2000]])
        pd.util.testing.assert_frame_equal(concat_df,
                                           lvl1.test_gdf[[1999, 2000]])

        load_map = lvl1.test_gdf.load(group=[1999, 2000], concat=False)
        for k, _df in load_map.items():
            pd.util.testing.assert_frame_equal(df_grps[k], _df)


        load_map = lvl1.test_gdf.load(group=None, concat=False)
        for k, _df in load_map.items():
            pd.util.testing.assert_frame_equal(df_grps[int(k)], _df)


        all_grp_df = lvl1.test_gdf.load(group=None, concat=True)
        pd.util.testing.assert_frame_equal(pd.concat(df_grps.values()),
                                           all_grp_df)


        html_str = lvl1.test_gdf._repr_html_()

        import numpy as np
        lvl1.test_gdf[2002] = df_grps[1999] * 3.3
        pd.util.testing.assert_frame_equal(lvl1.test_gdf[2002],
                                           df_grps[1999] *3.3)


        lvl1.test_gdf['a_series'] = df_grps[2001]['a']
        pd.util.testing.assert_series_equal(lvl1.test_gdf['a_series'],
                                            df_grps[2001]['a'])

    def test_pandas_sample(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        df = pd.DataFrame(dict(a=range(100), b=range(100, 200)))
        lvl1 = rt.mkrepo('lvl1')
        lvl1.save(df, name='test_df')
        s_df = lvl1.test_df.hdf.sample(n=5)
        pd.util.testing.assert_frame_equal(df.head(5), s_df)

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

    def test_keras_model_storage(self):
        import keras
        from keras.layers import Dense
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a')

        rt.subrepo_a.save('str object', name='some_data',
                          comments='something to search for')

        model = keras.models.Sequential()
        model.add(Dense(units=64, activation='relu',
                        input_dim=10))
        model.add(Dense(units=2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        rt.save(model, name='keras_model', author='unittest',
                input_data=rt.subrepo_a.some_data,
                features=['a', 'b'],
                target='not_real')
        self.assertTrue(rt.keras_model.storage_type == 'keras')

        tm = rt.keras_model.load()

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

    def test_sql_parameters(self):
        rt = idt.RepoTree(repo_root=self.repo_root_path)
        rt.mkrepo('subrepo_a').save('str object', name='some_data',
                                    comments='something to search for')

        sql_obj = idt.SQL(select_statement='id, name, age',
                          from_statement='db.table',
                          where_statement='age > {age_minimum}',
                          query_parameters=dict(age_minimum=0))

        rt.subrepo_a.save(sql_obj,
                          name='query_name_and_age')

        q = sql_obj.build_query()
        self.assertTrue('age_minimum' not in q)
        self.assertTrue('age > 0' in q)

        q = sql_obj.build_query(age_minimum=10)
        self.assertTrue('age_minimum' not in q)
        self.assertTrue('age > 10' in q)


# TODO:
# - Handle wrong types and check types within reason (e.g. strings!)
# - Basic search functionality off of tree
#   - Maybe use Wikipedia vocab to do n-gram similatiry between search and doc
#   - Do similarity between code cells?
#       - Writing something..."one of us has done this"...code simalrity
# - Test with Python2
if __name__ == '__main__':
    unittest.main()
