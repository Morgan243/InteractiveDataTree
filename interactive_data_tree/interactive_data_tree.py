# Author: Morgan Stuart
import math
import shutil
import abc
import string
import json
import pickle
import pandas as pd
import os
import time
import warnings
from datetime import datetime
from glob import glob
import sys

from .metadata import Metadata, standard_metadata
from .lockfile import LockFile
from .conf import *
from .utils import *

# - monkey patching docstrings
# - add a log of all operations in root
#   - Most recent (list most recently accessed/written)

# - Repo (dir) metadata - store create time of repo
# - Better support for dot paths (relative repo provided on save)
# - Enable references between objects
#   - Simply store a set of dot paths in the metadata
# - Log interface for easier adjustable verbosity and logging

def message_user(txt):
    print(txt)

def norm_vector(v):
    return math.sqrt(sum(_v**2.0 for _v in v))

# Same as dot
def inner_prod_vector(a, b):
    return sum(_a * _b for _a, _b in zip(a, b))

def cosine_sim(a, b):
    na = norm_vector(a)
    nb = norm_vector(b)
    sim = inner_prod_vector(a, b)/(na * nb)
    return sim

remove_chars = set(""",.!@#$%^&*()[]{}/\\`~-+|;:' \t\n\r""")
def clean_token(tk):
    return ''.join(c for c in tk if c not in remove_chars)

def basic_tokenizer(str_data, ngram_range=(1, 1)):
    ngrams = list(range(ngram_range[0], ngram_range[1] + 1))
    tokens = [s.lower() for s in str_data.split()]
    final_tokens = list()
    for n in ngrams:
        final_tokens += [clean_token(" ".join(tokens[i:i+n]))
                         for i in range(0, len(tokens), n)]
    return final_tokens


#####
def leaf_to_reference(obj, to_storage_interface=False):
    if isinstance(obj, basestring):
        return obj
    elif isinstance(obj, RepoLeaf):
        return obj.reference()
    elif isinstance(obj, StorageInterface):
        return obj.reference()
    else:
        return obj


class StorageInterface(object):
    """
    Base storage interface representing an arbitrary python object
    stored using pickle. New storage interfaces should be derived
    from this class.
    """
    storage_name = 'pickle'
    extension = 'pkl'
    expose_on_leaf = ['exists']
    interface_metadata = []
    required_metadata = []#['author']

    def __init__(self, parent_leaf):
        if not isinstance(parent_leaf, RepoLeaf):
            msg = "Parameter parent_leaf must be a RepoLeaf"
            msg += ", but got type '%s'" % str(type(parent_leaf))
            raise ValueError(msg)

        self.parent_leaf = parent_leaf
        self.name = parent_leaf.name
        self.path = parent_leaf.save_path + '.' + self.extension

        self.lock_file = self.path + '.' + idr_config['lock_extension']

        self.md_path = self.path + '.' + idr_config['metadata_extension']
        self.md = Metadata(path=self.md_path,
                           required_fields=self.required_metadata,
                           resolve_tree=self.parent_leaf.parent_repo)
        self.init()

    def __call__(self, *args, **kwargs):
        return self.load()

    def __str__(self):
        #return self.parent_leaf.reference(storage_type=self.storage_name)
        repos = self.parent_leaf.parent_repo.get_parent_repo_names()
        repos = ['ROOT'] + repos
        return ".".join(repos + [self.parent_leaf.name, self.storage_name])

    def get_associated_files_and_locks(self):
        return [(self.path, self.lock_file),
                (self.md.path, self.md.lock_path)]

    def init(self):
        pass

    def exists(self):
        return os.path.isfile(self.path)

    def md_exists(self):
        return self.md.exists()

    def load(self, **kwargs):
        """
        Locks the object and reads the data from the filesystem

        Returns
        -------
        Object stored
        """
        with LockFile(self.lock_file, lock_type='rlock'):
            with open(self.path, mode='rb') as f:
                obj = pickle.load(f)
            return obj

    def save(self, obj, **md_kwargs):
        """
        Locks the object and writes the object and metadata to the
        filesystem.

        Parameters
        ----------
        obj : Serializable Python object
        md_kwargs : Key-value pairs to include in the metadata entry

        Returns
        -------
        None
        """
        missing_md = self.md.get_missing_metadata_fields(md_kwargs, self.required_metadata)
        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        with LockFile(self.lock_file, lock_type='wlock'):
            with open(self.path, mode='wb') as f:
                pickle.dump(obj, f, protocol=2)

            self.write_metadata(obj=obj, user_md=md_kwargs)

    def reference(self):
        # Hack - References only at the leaf level for now
        return self.parent_leaf.reference()

    def write_metadata(self, obj=None, user_md=None, tree_md=None, si_md=None):
        """
        Locks metadata file, reads current contents, and appends
        md_kwargs key-value pairs to the metadata.

        Parameters
        ----------
        obj : object to which the metadata pertains
            If the object is provided, then the type name of
            the object can be stored in the metadata automatically.
            Derived classes can include other automatic metadata
            extraction (see HDF).
        md_kwargs : Key-value pairs to include in the metadata entry

        Returns
        -------
        None
        """
        #obj = dict() if obj is None else obj
        user_md = dict() if user_md is None else user_md
        tree_md = dict() if tree_md is None else tree_md
        si_md = dict() if si_md is None else si_md

        references = dict()
        for k in user_md.keys():
            # Overwrite leaf/URI to just URI
            user_md[k] = leaf_to_reference(user_md[k],
                                           to_storage_interface=True)

            # Normalize back to a leaf so we can update it's MD
            # store
            if is_valid_uri(user_md[k]):
                uri = user_md[k]
                if uri not in references:
                    references[uri] = list()

                references[uri].append(k)

        if obj is not None:
            tree_md['obj_type'] = type(obj).__name__

        cls = self.__class__
        tree_md['write_time'] = datetime.now().strftime(idr_config['date_format'])
        tree_md['md_update_time'] =tree_md['write_time']
        # TODO: This set math may be wrong or unecessary?
        tree_md['extra_metadata_keys'] = list(set(user_md.keys()) -
                                                 set(cls.required_metadata +
                                                     standard_metadata +
                                                     cls.interface_metadata))
        tree_md['references'] = references
        tree_md['md_vers'] = idr_config['md_vers']

        # TODO?
        # Remove this leaf from referrers to other
        # leafs if a reference was removed
        this_uri = self.parent_leaf.reference()
        for uri, keys in references.items():
            l = self.parent_leaf.parent_repo.from_reference(uri)

            if isinstance(l, RepoLeaf):
                # No type was specified, so reference the highest priority
                #self.md.add_referrer(this_uri, *keys)
                l.si.md.add_referrer(this_uri, *keys)
            elif isinstance(l, RepoTree):
                msg = ("Cannot reference a repository (%s), found in md keys: "
                       % l.name)
                msg += ", ".join(keys)
                raise ValueError(msg)

        md_kwargs = dict(user_md=user_md, tree_md=tree_md, si_md=si_md)

        self.md.write_metadata(**md_kwargs)

    def md_resolve(self, t_md):
        return md_resolve(self.parent_leaf.parent_repo, t_md=t_md)

    def read_metadata(self, most_recent=True, resolve_references=True,
                      user_md=True):
        """
        Read entire metadata history from storage. Each
        storage interface saves it's own metadata, so
        the interface must be specified, otherwise the
        priority order is used.

        Parameters
        ----------
        storage_type : str (default=None)
            Storage interface whose metadata to read

        Returns
        -------
        Metadata history as a list of dictionaries
        """
        md = self.md.read_metadata(most_recent=most_recent)
        if resolve_references:
            if most_recent:
                md['user_md'] = self.md_resolve(md.get('user_md', dict()))
                md['si_md'] = self.md_resolve(md.get('si_md', dict()))
            else:
                for _md in md:
                    _md['user_md'] = self.md_resolve(_md.get('user_md', dict()))
                    _md['si_md'] = self.md_resolve(_md.get('si_md', dict()))

        if user_md:
            md = md.get('user_md', dict()) if most_recent else [_md.get('user_md', dict()) for _md in md]

        return md


    def get_vector_representation(self):
        term_cnts = dict()
        terms = self.get_terms()
        for t in terms:
            term_cnts[t] = term_cnts.get(t, 0) + 1
        return term_cnts

    def get_terms(self):
        """
        Extract queryable plain text terms from the object. Defaults
        to returning metadata keys that are of str type.

        Returns
        -------
        List of string terms
        """
        all_md = self.md.read_metadata(most_recent=True)
        str_md_terms = list()
        for md in all_md.values():
            tmp_md_terms = [basic_tokenizer(v) for k, v in md.items()
                            if isinstance(v, basestring)]
            str_md_terms += [_v for v in tmp_md_terms for _v in v]
        return str_md_terms

    @staticmethod
    def _build_html_body_(md):
        tmd = md['tree_md']
        umd = md['user_md']
        html_str = """
        <b>Author</b>: {author} <br>
        <b>Last Write</b>: {ts} <br>
        <b>Comments</b>: {comments} <br>
        <b>Type</b>: {ty} <br>
        <b>Tags</b>: {tags} <br>
        """.format(author=umd.get('author'),
                   comments=umd.get('comments'), ts=tmd.get('write_time'),
                   ty=tmd.get('obj_type'), tags=umd.get('tags'))


        extra_keys = tmd.get('extra_metadata_keys', list())
        if len(extra_keys) > 0:
            html_items = ["<b>%s</b>: %s <br>" % (k, str(umd[k])[:140])
                          for k in extra_keys]
            add_md = """
            <h4>Additional Metadata</h4>
            %s
            """ % "\n".join(html_items)
            html_str += add_md

        return html_str

    @staticmethod
    def _get_two_column_div_template():
        div_template = """
        <div style="width: 80%;">
            <div style="float:left; width: 50%; height:100%; overflow: auto">
            {left_column}
            </div>
            <div style="float:right; width:49%; height: 100%; overflow: auto; margin-left:1%">
            {right_column}
            </div>
        </div>
        """
        return div_template

    def _repr_html_(self):
        md = self.read_metadata(resolve_references=False, user_md=False)
        html_str = """<h2> {name} </h2>""".format(name=self.name)
        html_str += StorageInterface._build_html_body_(md)
        return html_str


class HDFStorageInterface(StorageInterface):
    """
    Pandas storage interface backed by HDF5 (PyTables) for efficient
    storage of tabular data.
    """
    storage_name = 'hdf'
    extension = 'hdf'
    expose_on_leaf = ['sample'] + StorageInterface.expose_on_leaf
    hdf_data_level = '/data'
    hdf_format = 'fixed'
    #hdf_format = 'table'
    interface_metadata = ['length', 'columns', 'index_head']

    @staticmethod
    def __valid_object_for_storage(obj):
        return isinstance(obj, (pd.Series, pd.DataFrame, pd.Panel))

    def load(self):
        """
        Locks the object and reads the data from the filesystem

        Returns
        -------
        Object stored
        """
        with LockFile(self.lock_file, lock_type='rlock'):
            obj = pd.read_hdf(self.path, mode='r')
        return obj

    def save(self, obj, **md_kwargs):
        """
        Locks the object and writes the object and metadata to the
        filesystem.

        Parameters
        ----------
        obj : Serializable Python object
        md_kwargs : Key-value pairs to include in the metadata entry

        Returns
        -------
        None
        """
        if not self.__valid_object_for_storage(obj):
            raise ValueError("Expected Pandas Data object, got %s" % type(obj))

        with LockFile(self.lock_file, lock_type='wlock'):
            hdf_store = pd.HDFStore(self.path, mode='w')
            hdf_store.put(HDFStorageInterface.hdf_data_level,
                          obj, format=HDFStorageInterface.hdf_format)
            hdf_store.close()

        self.write_metadata(obj=obj, user_md=md_kwargs)

    # TODO: There are some bugs here for certain dataframe, some possibly related issues:
    # https://github.com/pandas-dev/pandas/pull/13267
    # https://github.com/pandas-dev/pandas/issues/11188
    # https://github.com/pandas-dev/pandas/issues/14568
    def sample(self, n=5):
        """
        [NOT A RANDOM SAMPLE] Fetch only the first N samples from
        the dataset

        Parameters
        ----------
        n : int
            Number of samples to retrieve

        Returns
        -------
        Pandas data object of the first n entries
        """
        with LockFile(self.lock_file, lock_type='rlock'):
            obj = pd.read_hdf(self.path, mode='r', stop=n)
        return obj

    def write_metadata(self, obj=None, user_md=None,
                       tree_md=None, si_md=None):
        """
        Locks metadata file, reads current contents, and appends
        md_kwargs key-value pairs to the metadata.


        Parameters
        ----------
        obj : object to which the metadata pertains
            If the object is provided, then additional
            metadata is automatically extracted:
                - Column names (if DataFrame)
                - First 5 index values
                - Number of samples

        md_kwargs : Key-value pairs to include in the metadata entry

        Returns
        -------
        None
        """

        if obj is not None and not self.__valid_object_for_storage(obj):
            raise ValueError("Expected Pandas Data object, got %s" % type(obj))

        si_md = dict()
        if isinstance(obj, pd.DataFrame):
            si_md['columns'] = list(str(c) for c in obj.columns)
            si_md['dtypes'] = list(str(d) for d in obj.dtypes)

        if obj is not None:
            si_md['index_head'] = list(str(i) for i in obj.index[:5])
            si_md['length'] = len(obj)

        super(HDFStorageInterface, self).write_metadata(obj=obj,
                                                        user_md=user_md,
                                                        si_md=si_md)

    def _repr_html_(self):
        md = self.read_metadata(user_md=False)

        basic_descrip = StorageInterface._build_html_body_(md)
        smd = md['si_md']

        extra_descrip = """
        <b>Num Entries </b> : {num_entries:,} <br>
        <b>Columns</b> ({n_cols:,}) : {col_sample} <br>
        <b>Index Head</b> : {ix_head} <br>
        """.format(num_entries=smd.get('length', -1),
                   n_cols=len(smd.get('columns', [])),
                   col_sample=", ".join(smd.get('columns', [])[:10]),
                   ix_head=", ".join(smd.get('index_head', [])))


        div_template = StorageInterface._get_two_column_div_template()
        div_template = div_template.format(left_column=basic_descrip,
                                           right_column=extra_descrip)

        html_str = """<h2> {name} </h2>""".format(name=self.name)
        html_str += div_template
        return html_str


class HDFGroupStorageInterface(HDFStorageInterface):
    storage_name = 'ghdf'
    extension = 'ghdf'
    expose_on_leaf = StorageInterface.expose_on_leaf
    hdf_data_level = '/idt_ghdf'
    hdf_format = 'fixed'
    #hdf_format = 'table'
    interface_metadata = ['length', 'columns', 'index_head']

    @staticmethod
    def __valid_object_for_storage(obj):
        #is_pd = isinstance(obj, (pd.Series, pd.DataFrame, pd.Panel))
        #return isinstance(obj, dict) or is_pd
        is_grp = isinstance(obj, pd.core.groupby.DataFrameGroupBy)
        valid_dict = isinstance(obj, dict) and all(isinstance(v, (pd.Series, pd.DataFrame))
                                                   for v in obj.values())
        return valid_dict or is_grp

    def __getitem__(self, item):
        return self.load(group=item)

    def __setitem__(self, key, value):
        kargs = {str(key):value}
        self.update(**kargs)

    def load(self, group=None, concat=True):
        with LockFile(self.lock_file, lock_type='rlock'):
            hdf_store = pd.HDFStore(self.path, mode='r')
            hdf_keys = hdf_store.keys()
            prefix = HDFGroupStorageInterface.hdf_data_level + '/'
            idt_groups = [k.replace(prefix, '') for k in hdf_keys]
            if hasattr(group, '__iter__') and not isinstance(group, basestring):
                if concat:
                    #ret = ret if not concat else pd.concat(ret.values())
                    ret = pd.concat(hdf_store.get(prefix + str(g))
                                    for g in group if str(g) in idt_groups)
                else:
                    ret = {g:hdf_store.get(prefix + str(g))
                           for g in group if str(g) in idt_groups}
            elif group is not None:
                ret = hdf_store[prefix + str(group)]
            else:
                if concat:
                    ret = pd.concat(hdf_store.get(g) for g in hdf_keys)
                else:
                    ret = {g:hdf_store.get(g) for g in hdf_keys}

            hdf_store.close()
        return ret

    def save(self, obj, **md_kwargs):
        """
        Locks the object and writes the object and metadata to the
        filesystem.

        Parameters
        ----------
        obj : Serializable Python object
        md_kwargs : Key-value pairs to include in the metadata entry

        Returns
        -------
        None
        """
        if not self.__valid_object_for_storage(obj):
            msg = "Expected Pandas DataFrame GroupBy object or Dict of pd objects, got %s" % type(obj)
            raise ValueError(msg)

        if isinstance(obj, dict):
            d_obj_iter = obj
        else:
            d_obj_iter = {gk: obj.get_group(gk) for gk in obj.groups.keys()}

        with LockFile(self.lock_file, lock_type='wlock'):
            hdf_store = pd.HDFStore(self.path, mode='a')
            try:
                from tqdm import tqdm
                with tqdm(total=len(d_obj_iter)) as pbar:
                    for k, v in d_obj_iter.items():
                        pbar.set_description("Storing group " + str(k))
                        hdf_p = HDFGroupStorageInterface.hdf_data_level + '/' + str(k)
                        hdf_store.put(hdf_p,
                                         v, format=HDFStorageInterface.hdf_format)
                        pbar.update(1)
            except ImportError:
                for k, v in d_obj_iter.items():
                    hdf_p = HDFGroupStorageInterface.hdf_data_level + '/' + str(k)
                    hdf_store.put(hdf_p,
                                     v, format=HDFStorageInterface.hdf_format)
            hdf_store.close()

        self.write_metadata(obj=d_obj_iter, user_md=md_kwargs)

    def update(self, **grps):
        if any(not isinstance(v, (pd.Series, pd.DataFrame))
               for v in grps.values()):
            msg = "All parameter values must be Pandas objects"
            raise ValueError(msg)

        d_grps = dict(grps)
        with LockFile(self.lock_file, lock_type='wlock'):
            hdf_store = pd.HDFStore(self.path, mode='a')
            for k, v in d_grps.items():
                hdf_p = HDFGroupStorageInterface.hdf_data_level + '/' + str(k)
                hdf_store.put(hdf_p,
                                 v, format=HDFStorageInterface.hdf_format)
            hdf_store.close()

    def write_metadata(self, obj=None, user_md=None,
                       tree_md=None, si_md=None):
        if obj is not None and not self.__valid_object_for_storage(obj):
            raise ValueError("Expected Dict of Pandas Data object, got %s" % type(obj))

        md = self.read_metadata(resolve_references=False, user_md=False)
        si_md = md.get('si_md', dict())
        current_groups = si_md.get('groups', dict())

        new_groups = dict()
        for g, o in obj.items():
            g_md = dict()
            if isinstance(o, pd.DataFrame):
                g_md['columns'] = list(str(c) for c in o.columns)
                g_md['dtypes'] = list(str(d) for d in o.dtypes)
                si_md['columns'] = list(set(si_md.get('columns', list()) + g_md['columns']))
                si_md['dtypes'] = list(set(si_md.get('dtypes', list()) + g_md['dtypes']))

            if o is not None:
                g_md['index_head'] = list(str(i) for i in o.index[:5])
                g_md['length'] = len(o)
                si_md['length'] = si_md.get('length', 0) + g_md['length']
            new_groups[g] = g_md

        current_groups.update(new_groups)
        si_md['groups'] = current_groups
        si_md['group_names'] = list(sorted(current_groups.keys()))

        StorageInterface.write_metadata(self,
                                        obj=None,
                                        tree_md=dict(obj_type='DataFrame Group'),
                                        user_md=user_md,
                                        si_md=si_md)

    def _repr_html_(self):
        md = self.read_metadata(user_md=False)

        basic_descrip = StorageInterface._build_html_body_(md)
        smd = md['si_md']
        groups = smd['groups']

        group_names = list(sorted(groups.keys()))
        if len(group_names) > 5:
            group_str = ", ".join(group_names[:2]) + " ... " + ", ".join(group_names[-2:])
        else:
            group_str = ", ".join(group_names)

        col_names = list(sorted(smd.get('columns', list())))
        if len(col_names) > 10:
            col_str = ", ".join(col_names[:5]) + " ... " + ", ".join(col_names[-5:])
        else:
            col_str = ", ".join(col_names)


        extra_descrip = """
        <b>Groups ({n_groups:,})</b> : {groups} <br>
        <b>Total Entries Across Groups </b> : {num_entries:,} <br>
        <b>Unique Columns</b> ({n_cols:,}) : {col_sample} <br>
        """.format(
            groups=group_str,
            n_groups=len(group_names),
            num_entries=smd.get('length', -1),
            n_cols=len(col_names),
            col_sample=col_str
        )


        div_template = StorageInterface._get_two_column_div_template()
        div_template = div_template.format(left_column=basic_descrip,
                                           right_column=extra_descrip)

        html_str = """<h2> {name} </h2>""".format(name=self.name)
        html_str += div_template
        return html_str


class NumpyStorageInterface(StorageInterface):
    storage_name = 'numpy'
    extension = 'npy'
    @staticmethod
    def __valid_object_for_storage(obj):
        import numpy as np
        return isinstance(obj, np.ndarray)

    def save(self):
        pass


class ModelStorageInterface(StorageInterface):
    storage_name = 'model'
    extension = 'mdl'
    expose_on_leaf = ['predict', 'predict_proba',
                      'features', 'target']
    required_metadata = ['input_data', 'features', 'target']
    def init(self):
        if not self.md_exists():
            return

        md = self.read_metadata()
        self.features = md['features']
        self.target = md['target']
        self.model = None

    def save(self, obj,
             **md_kwargs):

        missing_md = self.md.get_missing_metadata_fields(md_kwargs,
                                                         self.required_metadata)
        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        if isinstance(md_kwargs['input_data'], RepoLeaf):
            md_kwargs['input_data'] = md_kwargs['input_data'].reference()
        elif isinstance(md_kwargs['input_data'], StorageInterface):
            md_kwargs['input_data'] = md_kwargs['input_data'].parent_leaf.reference()

        super(ModelStorageInterface, self).save(obj=obj, **md_kwargs)

    def predict(self, X):
        if self.model is None:
            self.model = self.load()

        if isinstance(X, pd.DataFrame):
            _x = X[self.features].values
        else:
            _x = X
        preds = self.model.predict(_x)
        return preds

    def predict_proba(self, X):
        if self.model is None:
            self.model = self.load()

        if isinstance(X, pd.DataFrame):
            _x = X[self.features].values
        else:
            _x = X
        preds = self.model.predict_proba(_x)
        return preds


class KerasModelStorageInterface(ModelStorageInterface):
    storage_name = 'keras'
    extension = 'keras'
    expose_on_leaf = ModelStorageInterface.expose_on_leaf

    def save(self, model, **md_kwargs):
        import keras
        if not isinstance(model, keras.Model):
            msg = "KerasModelStorage expects a keras model"
            msg += ", instead got type %s" % (str(type(model)))
            raise ValueError(msg)

        missing_md = self.md.get_missing_metadata_fields(md_kwargs,
                                                         self.required_metadata)
        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        with LockFile(self.lock_file, lock_type='wlock'):
            model.save(self.path, overwrite=True)
            # TODO: Extract SI metadata infor about the model
            # e.g. number of layers, types of layers, in/put dims
            self.write_metadata(obj=model,
                                user_md=md_kwargs)

    def load(self, **kwargs):
        import keras
        with LockFile(self.lock_file, lock_type='rlock'):
            obj = keras.models.load_model(self.path)
        return obj


class SQL(object):
    def __init__(self, select_statement, from_statement,
                 where_statement='', query_parameters=None):
        self.select_statement = select_statement
        self.from_statement = from_statement
        self.where_statement = where_statement
        self.query_parameters = dict() if query_parameters is None else query_parameters
        self.section_params = dict()

        params_seen = set()
        params_expected = set(self.query_parameters.keys())
        for section, txt in self.asdict().items():
            # Lazy
            if section == 'query_parameters':
                continue
            parse_args = string.Formatter().parse(txt)
            self.section_params[section] = [pa[1] for pa in parse_args
                                if pa[1] is not None and isidentifier(pa[1])]
            self.section_params[section] = set(self.section_params[section])
            params_seen = params_seen.union(self.section_params[section])

        if params_expected != params_seen:
            msg = "Default values must be provided for all parameters, missing: "
            msg += ",".join(params_expected - params_seen)
            raise ValueError(msg)

    def asdict(self):
        return dict(select_statement=self.select_statement,
                    from_statement=self.from_statement,
                    where_statement=self.where_statement,
                    query_parameters=self.query_parameters)

    def build_query(self, **parameters):
        sel_p = {sp:parameters.get(sp, self.query_parameters.get(sp))
                    for sp in self.section_params['select_statement']}

        sel = self.select_statement.format(**sel_p)

        frm_p = {sp:parameters.get(sp, self.query_parameters.get(sp))
                 for sp in self.section_params['from_statement']}
        frm = self.from_statement.format(**frm_p)

        whr_p = {sp:parameters.get(sp, self.query_parameters.get(sp))
                 for sp in self.section_params['where_statement']}
        whr = self.where_statement.format(**whr_p)

        template = """
SELECT
  {select_statement}
FROM
  {from_statement}
  """.format(select_statement=sel,
             from_statement=frm)

        if self.where_statement.strip() != '':
            template += """
WHERE
  {where_statement}
        """.format(where_statement=whr)

        return template

    def _repr_html_(self):
        from pygments import highlight
        from pygments.lexers import SqlLexer
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(style='colorful')

        pygment_html = highlight(self.build_query(), SqlLexer(), formatter)
        style_html = """
        <style>
        {pygments_css}
        </style>
        """.format(pygments_css=formatter.get_style_defs())

        html = style_html + pygment_html
        return html


class SQLStorageInterface(StorageInterface):
    storage_name = 'sql'
    extension = 'sql'
    expose_on_leaf = ['query']
    required_metadata = []

    def load(self):
        with LockFile(self.lock_file, lock_type='rlock'):
            with open(self.path, mode='r') as f:
                obj = json.load(f)
        sql_obj = SQL(**obj)
        return sql_obj

    def save(self, obj, **md_kwargs):
        if isinstance(obj, dict):
            message_user("Converting dictionary to SQL object")
            obj = SQL(**obj)
        elif not isinstance(obj, SQL):
            msg = "SQLStorage expects a datatree SQL object, but got %s"
            raise ValueError(msg % str(type(obj)))

        missing_md = self.md.get_missing_metadata_fields(md_kwargs, self.required_metadata)

        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        store_dict = dict(obj.asdict())
        with LockFile(self.lock_file, lock_type='wlock'):
            with open(self.path, mode='w') as f:
                json.dump(store_dict, f)

            self.write_metadata(obj=obj, user_md=md_kwargs)

    def query(self, cxn, **parameters):
        if not hasattr(cxn, 'query'):
            msg = 'Argument cxn must have a query method'
            raise ValueError(msg)

        q = self.load().build_query(**parameters)
        #q = SQLStorageInterface.build_query(self.load())
        return cxn.query(q)

    def _repr_html_(self):
        md = self.read_metadata(user_md=False)

        basic_descrip = StorageInterface._build_html_body_(md)
        #sql_txt = SQLStorageInterface.build_query(self.load())
        sql_txt = self.load().build_query()

        from pygments import highlight
        from pygments.lexers import SqlLexer
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(style='colorful')
        pygment_html = highlight(sql_txt, SqlLexer(), formatter)
        style_html = """
        <style>
        {pygments_css}
        </style>
        """.format(pygments_css=formatter.get_style_defs())

        div_template = StorageInterface._get_two_column_div_template()
        div_template = div_template.format(left_column=basic_descrip,
                                           right_column=style_html + pygment_html)

        html_str = """<h2> {name} </h2>""".format(name=self.name)
        html_str += div_template
        return html_str

#######


#######
# Data structures to hold and map interfaces with names/extensions
storage_interfaces = dict()

extension_to_interface_name_map = dict()

type_storage_lookup = dict()

storage_type_priority_order = list()
storage_type_priority_map = dict()

def register_storage_interface(interface_class, name,
                               priority=None, types=None):
    global storage_type_priority_map
    global storage_type_priority_order
    global type_storage_lookup
    global extension_to_interface_name_map
    global storage_interfaces

    if name in storage_interfaces:
        raise ValueError("Storage interfaced named '%s' is already registered!" % name)

    if not issubclass(interface_class, StorageInterface):
        args = (name, str(interface_class), str(StorageInterface))
        raise ValueError("Storage interfaced named '%s' of type %s is not a subclass of %s" % args)

    if interface_class.extension in extension_to_interface_name_map:
        msg = "%s has specified the extension '%s', which is already in use by %s"
        raise ValueError(msg % (str(interface_class),
                                interface_class.extension,
                                extension_to_interface_name_map[interface_class.extension]))

    storage_interfaces[name] = interface_class
    extension_to_interface_name_map[interface_class.extension] = name

    if priority is None:
        storage_type_priority_order.append(name)
        storage_type_priority_map[name] = len(storage_type_priority_map) - 1
    else:
        for k in storage_type_priority_map.keys():
            # Shift lower and equal priorities down priorities
            if storage_type_priority_map[k] >= priority:
                storage_type_priority_map[k] += 1

        storage_type_priority_map[name] = priority
        rev_map = {v:k for k, v in storage_type_priority_map.items()}
        storage_type_priority_order = [rev_map[i]
                                       for i in range(len(storage_type_priority_map))]


    if types is not None:
        if not isinstance(types, list):
            types = [types]
        type_storage_lookup.update({t:name for t in types})

def storage_type_for_obj(obj, default='pickle'):
    # TODO: Return priority order if multiples match
    for ty, si_name in type_storage_lookup.items():
        if isinstance(obj, ty):
            return si_name
    return default


register_storage_interface(HDFStorageInterface, 'hdf', 0,
                           types=[pd.DataFrame, pd.Series])

register_storage_interface(StorageInterface, 'pickle', 1)
register_storage_interface(ModelStorageInterface, 'model', 2)
register_storage_interface(SQLStorageInterface, 'sql', 3,
                           types=[SQL])
register_storage_interface(HDFGroupStorageInterface, 'ghdf', 4,
                           types=[pd.core.groupby.DataFrameGroupBy])
try:
    import keras
    register_storage_interface(KerasModelStorageInterface, 'keras', 5,
                               types=[keras.Model, keras.Sequential])
except ImportError:
    #raise
    print("Keras cannot be imported - its storage interface will be unavailable")

#def is_valid_idt_from_fname(fname):
#    s = fname.split('.')
#    if len(s) == 1:
#        pass
#    elif len(s) == 2:
#        maybe_o_name, maybe_o_type = s
#        if maybe_o_type in storage_interfaces:
#            return True
#    elif len(s) == 3:
#        # Possible Lock file
#        pass
#
#    return False


class RepoLeaf(object):
    """
    A terminating node in the repo tree, providing an interface
    to possibly several objects saved using different storage types.
    """
    # ABC used in order to allow doc string monkey patching
    __metaclass__ = abc.ABCMeta

    def __init__(self, parent_repo, name, fnames=None,
                 update_doc_strings=False):
        """
        Parameters
        ----------
        parent_repo : RepoTree object
            RepoTree in which this leaf resides
        name : str
            Name of the object
        """
        self.parent_repo = parent_repo
        self.name = name

        self.save_path = os.path.join(self.parent_repo.idr_prop['repo_root'], self.name)
        self.type_to_storage_interface_map = dict()
        self.si = None
        self.storage_type = None
        self.update_doc_strings = update_doc_strings
        self.refresh(fnames=fnames)

    def __call__(self, *args, **kwargs):
        """
        Load the object stored under this name, uses
        storage interface priority list.

        Returns
        -------
        Stored data object

        """
        return self.load()

    def __getitem__(self, item):
        if hasattr(self.si, '__getitem__'):
            return self.si[item]
        else:
            msg = "Storage interface '%s' has setitem implementation" % self.storage_type
            raise NotImplementedError(msg)

    def __setitem__(self, key, value):
        if hasattr(self.si, '__setitem__'):
            self.si[key] = value
        else:
            msg = "Storage interface '%s' has setitem implementation" % self.storage_type
            raise NotImplementedError(msg)

    def __str__(self):
        return self.reference()

    def __update_doc_str(self):
        docs = self.name + "\n\n"
        #si = self._get_highest_priority_si()
        si = self.si

        if si is None:
            return

        md = si.read_metadata(resolve_references=False)

        auth = md.get("author")
        comm = md.get("comments")
        ts = md.get("write_time")
        ty = md.get('obj_type')
        tags = md.get('tags')

        docs += si.name + "\n" + "-"*len(si.name) + "\n"
        docs += "  Author: " + (auth if auth is not None else "No author") + "\n"
        docs += "  Timestamp: " + (ts if ts is not None else "No timestamp") + "\n"
        docs += "  Comments: " + (comm if comm is not None else "No comments") + "\n"
        docs += "  Type: " + (ty if ty is not None else "No type") + "\n"
        docs += "  Tags: " + (tags if tags is not None else "No tags") + "\n"
        docs += "\n\n"
        self.__doc__ = docs

    def _repr_html_(self):
        return self.si._repr_html_()

    def reference(self):
        r_names = self.parent_repo.get_parent_repo_names()
        r_names.append(self.parent_repo.name)
        r_names.append(self.name)
        ref_str = '/'.join(r_names)
        ref_str = URI_SPEC + ref_str
        return ref_str

    def refresh(self, fnames=None):
        mde = idr_config['metadata_extension']
        repe = idr_config['repo_extension']

        if fnames is None:
            fnames = [p for p in glob(self.save_path + '.*')
                            if p[-len(mde):] != mde
                            and p[-len(repe):] != repe]

        if len(fnames) == 1:
            if self.si is None:
                path = fnames[0]
                fname = os.path.split(path)[-1]
                name, ext = fname.split('.')
                si_name = extension_to_interface_name_map[ext]
                si_cls = storage_interfaces[si_name]
                self.si = si_cls(parent_leaf=self)

            setattr(self, self.si.storage_name, self.si)
            self.md = self.si.md
            for l in self.si.expose_on_leaf:
                setattr(self, l, getattr(self.si, l, None))

        elif len(fnames) > 1:
            msg = "to many files: %s" % "\n".join(fnames)
            raise ValueError(msg)

        if self.update_doc_strings:
            self.__update_doc_str()
        return self

    def save(self, obj, storage_type=None, auto_overwrite=False,
             verbose=True, **md_props):
        """
        Save the object and metadata to the filesystem using a specific
        storage interface

        Parameters
        ----------
        obj : Python object to save
            The python object must be compatable with the
            selected storage_type
        storage_type : str (default=None)
            Storage interface type name to use
        auto_overwrite : bool (default=False)
            If True, user is not prompted for overwriting an existing object.
        verbose : bool (default=True)
            If True, prints additional debug and feedback messages

        md_kwargs : Key-value pairs to include in the metadata entry
            Some storage interfaces may automatically extract metadata

        Returns
        -------
        None
        """
        # Need target save type for conflict detection and the eventual save
        if storage_type is None:
            storage_type = storage_type_for_obj(obj, default='pickle')
        elif storage_type not in storage_interfaces:
            msg = "Unknown storage type '%s', expecting one of %s"
            msg = msg % (storage_type, ",".join(storage_interfaces.keys()))
            raise ValueError(msg)

        if self.storage_type != storage_type:
            msg = ("You're trying to save a leaf of type '%s', but this leave is '%s'"
                   %(storage_type, self.storage_type))
            msg += "\nDelete this leaf first or change to matching storage type"

        last_storage_type = self.storage_type
        self.storage_type = storage_type

        # Construct the filesystem path using a typed extension
        self.si = storage_interfaces[storage_type](parent_leaf=self)

        # Double Check - file exists there or file is registered in memory
        if self.si.exists():
            if auto_overwrite and verbose:
                print("Auto overwriting '%s' (%s) in %s " % (self.name,
                                                             storage_type,
                                                             self.parent_repo.name))
            elif not auto_overwrite:
                prompt = "An object named '%s' (%s) in %s already exists" % (self.name,
                                                                             storage_type,
                                                                             self.parent_repo.name)
                prompt += "\nOverwrite this object? (y/n)"
                try:
                    y = raw_input(prompt)
                except NameError:
                    y = input(prompt)

                if y == 'y' or y == 'yes':
                    print("Proceeding with overwrite...")
                else:
                    print("Aborting...")
                    return

        if verbose:
            message_user("Saving to: %s.%s (%s)" % (self.parent_repo.name,
                                                    self.name, storage_type))

        self.si.save(obj, **md_props)

        if hasattr(self.si, 'init'):
            self.si.init()

        if last_storage_type != self.storage_type and last_storage_type is not None:
            delattr(self, last_storage_type)

        #setattr(self, self.storage_type, self.si)

        if verbose:
            message_user("Save Complete")
            message_user("-------------")

        #self.parent_repo._append_to_master_log(operation='save', leaf=self,
        #                                       author=md_props.get('author', None))

        #self.parent_repo._add_to_index(leaf=self)
        self.parent_repo.write_hooks(self)

        self.refresh()

    # TODO: Move delete to storage interface, perform with lock?
    def delete(self, author):
        """
        Permanently destroy the object

        Parameters
        ----------
        storage_type : str (default=None)
            Storage interface type name to use

        Returns
        -------
        None
        """
        filenames = glob(self.save_path + '.*')
        message_user("Deleting:\n%s" % "\n".join(filenames))
        [os.remove(fn) for fn in filenames]

        #self.parent_repo._append_to_master_log(operation='delete', leaf=self,
        #                                       author=author)
        #self.parent_repo._remove_from_index(self, missing_err='ignore')
        self.parent_repo.delete_hooks(self)
        self.parent_repo.refresh()

    def rename(self, new_name, author):
        if not isidentifier(new_name):
            msg = "New name must be a valid python identifier, got '%s'" % new_name
            raise ValueError(msg)

        if new_name in self.parent_repo.list(list_repos=False):
            msg = ("Repo '%s' already has a leaf with name '%s'"
                   % (self.parent_repo.name, new_name))
            raise ValueError(msg)

        orig_si_ref = self.reference()

        orig_name = self.name
        base_p = self.parent_repo.idr_prop['repo_root']
        files_and_locks = self.si.get_associated_files_and_locks()
        # Remove all files using associated lock files
        for f, l_f in files_and_locks:
            # Will break if lock file is not valid...
            with LockFile(l_f, lock_type='wlock'):
                fname = os.path.split(f)[-1]
                new_fname = fname.replace(orig_name, new_name)
                new_p = os.path.join(base_p, new_fname)
                shutil.move(f, new_p)

        #self.parent_repo._append_to_master_log(operation='rename',
        #                                       leaf=self,
        #                                       author=author)

        try:
            # Remove storage type from the index
            self.parent_repo.delete_hooks(self)
        except ValueError as e:
            ref = self.reference()
            message_user("Unable to execute hooks on leaf: '%s'" % ref)

        self.parent_repo.write_hooks(self)
        self.parent_repo.refresh()
        self.parent_repo[new_name].update_references(orig_si_ref)

    def move(self, tree, author):
        """
        Move this leaf to RepoTree provided as the 'tree' argument.
        The underlying file contents are moved on the filesystem,
        the index is updated, and the tree is refreshed

        Parameters
        ----------
        tree : RepoTree object
            Destination repo location as a RepoTree object
        author: string
            User/system performing the action

        Returns
        -------
            None
        """

        if self.name in tree.list(list_repos=False):
            msg = ("Repo '%s' already has a leaf with name '%s'"
                   % (tree.name, self.name))
            raise ValueError(msg)

        new_base_p = tree.idr_prop['repo_root']
        #orig_si_uris = {ty:si.reference()
        #                for ty, si in self.type_to_storage_interface_map.items()}
        #orig_si_ref = self.si.reference()
        orig_si_ref = self.reference()

        files_and_locks = self.si.get_associated_files_and_locks()
        # Remove all files using associated lock files
        for f, l_f in files_and_locks:
            # Will break if lock file is not valid...
            with LockFile(l_f, lock_type='wlock'):
                fname = os.path.split(f)[-1]
                new_p = os.path.join(new_base_p, fname)
                shutil.move(f, new_p)

        # Remove storage type from the index
        self.parent_repo.delete_hooks(self)
        self.parent_repo.refresh()
        tree.refresh()
        #self.parent_repo._append_to_master_log(operation='move', leaf=self,
        #                                       author=author,
        #                                       storage_type=self.si.storage_name)

        #self.parent_repo._add_to_index(tree[self.name])
        self.parent_repo.write_hooks(tree[self.name])


        tree[self.name].update_references(orig_si_ref)

    def load(self, **kwargs):
        """
        Load the object from the filesystem

        Parameters
        ----------
        storage_type : str (default=None)
            Storage interface type to load from

        Returns
        -------
        Stored Object
        """
        return self.si.load(**kwargs)

    def update_references(self, previous_si_uri, delete=False):
        # Go through new interfaces, make sure that any interfaces that
        # were referring to this interface before move are updated
        #for ty, si in self.type_to_storage_interface_map.items():

        md = self.si.read_metadata(resolve_references=False, user_md=False)
        # If self has been deleted, we only need to remove references/referrers
        # so we shouldn't try and get the current URI since it is irrelevant
        if not delete:
            new_si_uri = self.si.reference()

        referrers = md['tree_md'].get('referrers', dict())
        for referrer_uri, keys in referrers.items():
            # 'l' is a leaf that references this leaf
            # -> Since this leaf is moving, we must update l's references
            #l = reference_to_leaf(self.parent_repo, referrer_uri)
            l = self.parent_repo.from_reference(referrer_uri)
            assert isinstance(l, RepoLeaf)
            l.si.md.remove_reference(previous_si_uri)
            if not delete:
                l.si.md.add_reference(new_si_uri, *keys)

        references = md['tree_md'].get('references', dict())
        for reference_uri, keys in references.items():
            # 'l' is a leaf that this leaf references
            # -> Since this leaf is moving, we must update l's referrers
            #l = reference_to_leaf(self.parent_repo, reference_uri)
            l = self.parent_repo.from_reference(reference_uri)
            l.si.md.remove_referrer(previous_si_uri)
            if not delete:
                l.si.md.add_referrer(new_si_uri, *keys)

    def read_metadata(self, most_recent=True, resolve_references=True,
                      user_md=True):
        return self.si.read_metadata(most_recent=most_recent,
                                     resolve_references=resolve_references,
                                     user_md=user_md)

class RepoTree(object):
    """
    A branching node in the repo tree, containing both objects
    and other sub-repositories.
    """
    def __init__(self, repo_root=None, parent_repo=None):
        """
        Parameters
        ----------
        repo_root : Directory that the repo is in (parent dir)
        parent_repo : Repo Tree Object representing the parent repo
        """
        if repo_root is None:
            repo_root = idr_config['storage_root_dir']

        if repo_root[-len(idr_config['repo_extension']):] != idr_config['repo_extension']:
            repo_root = repo_root + '.' + idr_config['repo_extension']

        if not os.path.isdir(repo_root):
            os.mkdir(repo_root)

        self.idr_prop = dict(
            repo_root=repo_root,
            parent_repo=parent_repo,
        )
        self.idr_prop['repo_name'] = os.path.split(repo_root)[-1].replace('.' + idr_config['repo_extension'], '')
        self.name = self.idr_prop['repo_name']
        self.__repo_object_table = dict()
        self.__sub_repo_table = dict()

        index_f = lambda lf: self._add_to_index(lf)
        self.__write_hooks = dict(index=index_f)

        del_index_f = lambda lf: self._remove_from_index(lf)
        self.__delete_hooks = dict(index=del_index_f)

        self.refresh()

    def __contains__(self, item):
        try:
            t = self[item]
            return True
        except KeyError as e:
            return False

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self[_item] for _item in item]

        in_repos = item in self.__sub_repo_table
        in_objs = item in self.__repo_object_table

        if in_objs and in_repos:
            #raise ValueError("Somehow this item is in both obj and repos!?>")
            # If in both, return the tree
            return self.__sub_repo_table[item]
        elif in_repos:
            return self.__sub_repo_table[item]
        elif in_objs:
            return self.__repo_object_table[item]
        else:
            #try:
                #ret = getattr(self, item)
            raise KeyError("%s is not in the tree" % str(item))

    def __setitem__(self, key, value):
        raise NotImplementedError("How this should behave is not quite clear yet :/")

    def __getattr__(self, item):

        # Only called if an unknown attribute is accessed
        # - if so, then check that the object wasn't created by another instance
        # i.e. rescan repo before throwing an error

        do_refresh = item not in self.__repo_object_table and item not in self.__sub_repo_table

        if do_refresh:
            self.refresh()

        if item in self.__repo_object_table:
            return self.__repo_object_table[item]
        elif item in self.__sub_repo_table:
            return self.__sub_repo_table[item]
        else:
            dot_path = ".".join(self.get_parent_repo_names())
            dot_path = "root" if len(dot_path) == 0 else dot_path
            raise AttributeError("'%s' is not under repo %s" % (item, dot_path))

    def __len__(self):
        return (len(self.__repo_object_table)
                + len(self.__sub_repo_table))

    def __update_doc_str(self):
        docs = "Repository Name: " + self.name + "\n\n"

        sub_repos = list(self.__sub_repo_table.keys())
        repo_objs = list(self.__repo_object_table.keys())

        if len(sub_repos) > 0:
            sr_str = "\n".join("%s (%d)" % (sr_name, len(sr.list()))
                                for sr_name, sr in self.__sub_repo_table.items())
        else:
            sr_str = "No sub-repositories"

        if len(repo_objs) > 0:
            ro_str = "\n".join("%s [%s]" % (ro_name, ", ".join(ro.type_to_storage_interface_map))
                               for ro_name, ro in self.__repo_object_table.items())
        else:
            ro_str = "No objects stored"

        d_str = """Objects in Repo
----------------
{ro}

Sub-Repositories
----------------
{sr}\n\n""".format(ro=ro_str, sr=sr_str)

        self.__doc__ = docs + d_str

    def _load_master_log(self):
        root_repo = self.get_root()
        log_name = idr_config['master_log']
        log_exists = log_name in root_repo.list(list_repos=False)
        if not log_exists:
            message_user("Log doesn't exist yet, creating it at %s.%s" % (self.name,
                                                                          log_name) )
            root_repo.save([], name=log_name, author='system',
                           comments='log of events across entire tree',
                           verbose=False,
                           tags='idt_log')

        return root_repo.load(name=log_name)

    def _write_master_log(self, log_data):
        root_repo = self.get_root()
        log_name = idr_config['master_log']

        root_repo.save(log_data, name=log_name, author='system',
                       comments='log of events across entire tree',
                       verbose=False,
                       tags='log', auto_overwrite=True)

    def _append_to_master_log(self, operation,
                              leaf=None, storage_type=None,
                              author=None):
        # Don't need to index the LOG itself
        if leaf is not None:
            if leaf.name in (idr_config['master_log'], idr_config['master_index']):
                return

        log_data = self._load_master_log()

        entry = dict(base_master_log_entry)
        entry['repo_tree'] = self.get_parent_repo_names() + [self.name]
        entry['repo_leaf'] = None if leaf is None else leaf.name
        entry['storage_type'] = storage_type
        entry['repo_operation'] = operation
        entry['timestamp'] = datetime.now()
        entry['cwd'] = os.getcwd()
        entry['nb_name'] = shared_metadata.get('notebook_name')
        entry['author'] = author if author is not None else shared_metadata.get('author')

        log_data.append(entry)
        self._write_master_log(log_data)

    def _ipython_key_completions_(self):
        k = list(self.__sub_repo_table.keys())
        k += list(self.__repo_object_table.keys())
        return k

    def _repr_html_(self):
        repos_html = """
        <h4>Sub Repos</h4>
        %s
        """ % "\n".join("<li>%s [%d]</li>" % (rt, len(self.__sub_repo_table[rt]))
                        for rt in sorted(self.__sub_repo_table.keys()))
        objects_html = """
        <h4>Objects</h4>
        %s
        """ % "\n".join("<li>%s [%s]</li>" %
                        (rt, self.__repo_object_table[rt].si.storage_name)
                        for rt in sorted(self.__repo_object_table.keys()))

        if self.idr_prop['parent_repo'] is not None:
            parent_repo_str = "->".join(['Root']
                                        + self.get_parent_repo_names()[1:]
                                        + [self.name])
        else:
            parent_repo_str = "Root (%s)" % self.name
        html = """
        {repo_parent_header}
        <div style="width: 75%;">
            <div style="float:left; width: 50%; height:100%; overflow: auto">
                {repos_list}
            </div>
            <div style="float:right; width:49%; height: 100%; overflow: auto; margin-left:1%">
                {objs_list}
            </div>
        </div>
        """.format(repo_parent_header=parent_repo_str,
                   repos_list=repos_html, objs_list=objects_html)

        return html

    def _load_master_index(self):
        # Open index object in root tree
        #   - Each type has its own index, so use storage type to distinguish
        root_repo = self.get_root()
        ix_name = idr_config['master_index']
        ix_exists = ix_name in root_repo.list(list_repos=False)
        if not ix_exists:
            message_user("Master index doesn't exists yet, creating it at %s.%s"
                         % (root_repo.name, ix_name))
            #root_repo.save(dict(), name=ix_name, author='system',
            #               comments='index of objects across entire tree',
            #               tags='idt_index', verbose=False, auto_overwrite=True)
            self._write_master_index(dict())
        return root_repo.load(name=ix_name, storage_type='pickle')

    def _write_master_index(self, index):
        root_repo = self.get_root()
        ix_name = idr_config['master_index']
        #ix_exists = ix_name in root_repo.list(list_repos=False)
        root_repo.save(index, name=ix_name, storage_type='pickle',
                       author='system',
                       comments='index of objects across entire tree',
                       tags='idt_index', auto_overwrite=True,
                       verbose=False)

    def _remove_from_index(self, leaf, missing_err='raise'):
        if missing_err not in ('ignore', 'raise'):
            raise ValueError("Unknown value for param missing_err, expected one of 'raise', 'ignore'")

        master_index = self._load_master_index()
        ref = leaf.reference()
        if ref in master_index:
            del master_index[ref]
        elif missing_err == 'raise':
            raise ValueError("'%s' is not in index" % ref)

        self._write_master_index(master_index)

    def _add_to_index(self, leaf):
        # Avoid indexing the index
        if leaf.name in (idr_config['master_log'], idr_config['master_index']):
            return

        # Open index object in root tree
        #   - Each type has its own index, so use storage type to distinguish
        vec_map = leaf.si.get_vector_representation()
        master_index = self._load_master_index()
        ref = leaf.reference()
        master_index[ref] = vec_map
        self._write_master_index(master_index)

    def write_hooks(self, leaf):
        for h_name, h in self.__write_hooks.items():
            h(leaf)

    def delete_hooks(self, leaf):
        for h_name, h in self.__delete_hooks.items():
            h(leaf)

    def search(self, q_str, top_n=5, interactive=True):
        # TODO: add in filters
        # - storage type
        # - metadata key substr match
        q_tokens = basic_tokenizer(str_data=q_str)
        q_grams = dict()
        for tk in q_tokens:
            q_grams[tk] = q_grams.get(tk, 0) + 1

        ix = self._load_master_index()

        # Parse a lexicon from existing vectors
        lex = set([tk for vec_dict in ix.values()
                   for tk in vec_dict.keys()]
                  + list(q_grams.keys()))
        lex_d = {tk:0 for tk in lex}

        q_vec = dict(lex_d)
        q_vec.update(q_grams)
        q_vec = [q_vec[k] for k in sorted(lex)]

        sim_res = dict()
        for leaf_path, vec_dict in ix.items():
            tmp_vec = dict(lex_d)
            tmp_vec.update(vec_dict)
            tmp_vec = [tmp_vec[k] for k in sorted(lex)]
            sim_res[leaf_path] = cosine_sim(q_vec, tmp_vec)

        s_keys = sorted(sim_res.keys(),
                        key=lambda k: sim_res[k],
                        reverse=True)

        if top_n is not None:
            sim_res = {k: sim_res[k] for k in s_keys}

        if interactive:
            for i, k in enumerate(s_keys):
                print("[%d] %s : %.2f" % (i, k, sim_res[k]*100.0))
        else:
            return sim_res

    def get_root(self):
        """
        Returns
        -------
        Path (str) to the repo root (top-level repo)
        """
        rep = self
        while rep.idr_prop['parent_repo'] is not None:
            rep = rep.idr_prop['parent_repo']
        return rep

    def get_parent_repo_names(self):
        """
        Returns
        -------
        List of repo names (str) in hierarchical order
        """
        rep = self
        repos = list()
        while rep.idr_prop['parent_repo'] is not None:
            rep = rep.idr_prop['parent_repo']
            repos.append(rep.name)

        return list(reversed(repos))

    def refresh(self, all_dir_items=None):
        """
        Force the tree to rebuild by rescanning the filesystem

        Returns
        -------
        self
        """
        mde = idr_config['metadata_extension']
        repe = idr_config['repo_extension']
        locke = idr_config['lock_extension']
        repo_curr = dict(self.__sub_repo_table)
        obj_curr = dict(self.__repo_object_table)
        self.__sub_repo_table = dict()
        self.__repo_object_table = dict()

        if all_dir_items is None:
            all_dir_items = os.listdir(self.idr_prop['repo_root'])

        matching_items = [f for f in all_dir_items
                          if ('.' in f) and f[-len(mde):] != mde]
        repo_items = [f for f in matching_items
                          if f[-len(repe):] == repe]
        obj_items = [f for f in matching_items
                      if f[-len(repe):] != repe
                      and f[-len(locke):] != locke]

        for rf in repo_items:
            r = rf.split('.')
            if len(r) == 2:
                base_name, ext = r
            elif len(r) > 2:
                # Assume the repo name has a period?
                raise ValueError("Repository name cannot have a '.': %s" % rf)

            if base_name in repo_curr:
                self.add_repo_tree(repo_curr[base_name].refresh())
            else:
                p = os.path.join(self.idr_prop['repo_root'], rf)
                self.add_repo_tree(RepoTree(repo_root=p, parent_repo=self))

        for of in obj_items:
            r = of.split('.')
            if len(r) == 2:
                base_name, ext = r
            elif len(r) > 2:
                # Assume the repo name has a period?
                raise ValueError("Object name cannot have a '.': %s" % of)

            obj_files = list(filter(lambda s: (base_name == s.split('.')[0]), obj_items))
            # Had the object before - don't create new Leaf
            if base_name in obj_curr:
                self.add_obj_leaf(obj_curr[base_name].refresh(fnames=obj_files))
            else:
                self.add_obj_leaf(RepoLeaf(parent_repo=self, name=base_name, fnames=obj_files))

        curr_names = set(list(obj_curr.keys()) + list(repo_curr.keys()))
        new_names = set(list(self.__repo_object_table.keys()) + list(self.__sub_repo_table.keys()))
        for n in curr_names - new_names:
            delattr(self, n)

        return self

    def delete(self, name, author):
        """
        Permanently destroy the object

        Parameters
        ----------
        name : str
            Name of the object in this repo to delete
        storage_type : str (default=None)
            Storage interface type name to use

        Returns
        -------
        None
        """
        self.__repo_object_table[name].delete(author=author)
        if hasattr(self, name) and isinstance(getattr(self, name), RepoLeaf):
            delattr(self, name)

        self.__update_doc_str()

    def move(self, name, tree, author):
        """
        Move an object to RepoTree provided as the 'tree' argument.
        The underlying file contents are moved on the filesystem,
        the index is updated, and the tree is refreshed

        Parameters
        ----------
        name : string
            Name of repo object to move
        tree : RepoTree object
            Destination repo location as a RepoTree object
        author: string
            User/system performing the action

        Returns
        -------
            None
        """

        if name in tree.list(list_repos=False):
            msg = "Repo '%s' already has a leaf with name '%s'" % (tree.name, name)
            raise ValueError(msg)

        self.__repo_object_table[name].move(tree, author=author)

    def add_obj_leaf(self, leaf):
        self.__repo_object_table[leaf.name] = leaf

        repo_coll = leaf.name in self.__sub_repo_table

        if not repo_coll:
            setattr(self, leaf.name, leaf)

        self.__update_doc_str()

    def add_repo_tree(self, tree):
        self.__sub_repo_table[tree.name] = tree
        setattr(self, tree.name, tree)
        self.__update_doc_str()

    def save(self, obj, name, author=None, comments=None, tags=None,
             auto_overwrite=False, storage_type=None, **extra_kwargs):
        """
        Save the object and metadata to the filesystem.

        Parameters
        ----------
        obj : Python object to save
            The python object must be compatable with the
            selected storage_type
        name : str
            Name of the object/leaf to create. Must be a valid
            Python name.
        author : str (default=None)
            Name or ID of the creator
        comments : str (default=None)
            Misc. comments to include in the metadata
        tags : str (default=None)
            Whitespace separated tags to include in the metadata
        auto_overwrite : bool (default=False)
            If True, user is not prompted for overwriting an existing object.
        storage_type : str (default=None)
            Storage interface type name to use
        md_kwargs : Key-value pairs to include in the metadata entry
            Some storage interfaces may automatically extract metadata

        Returns
        -------
        None
        """
        if not isidentifier(name):
            raise ValueError("Name must be a valid python identifier, got '%s'" % name)

        # These are special metadata keys that are built-in and exposed as args
        if author is not None:
            extra_kwargs['author'] = author
        if comments is not None:
            extra_kwargs['comments'] = comments
        if tags is not None:
            extra_kwargs['tags'] = tags

        leaf = self.__repo_object_table.get(name,
                                            RepoLeaf(parent_repo=self, name=name))
        leaf.save(obj, storage_type=storage_type,
                  auto_overwrite=auto_overwrite,
                  **extra_kwargs)
        self.add_obj_leaf(leaf)

    def load(self, name, **kwargs):
        """
        Load the object from the filesystem

        Parameters
        ----------
        name : str
            Name of the object to load
        storage_type : str (default=None)
            Storage interface type to load from

        Returns
        -------
        Stored Object
        """
        if name not in self.__repo_object_table:
            raise ValueError("Unknown object %s in repo %s" % (name, self.name))

        #return self.__repo_object_table[name].load()
        st = self.__repo_object_table[name]
        return st.load(**kwargs)

    def mkrepo(self, name, err_on_exists=False):
        """
        Create a new repo off this tree

        Parameters
        ----------
        name : str
            Name of the repository. Must be a valid python name
        err_on_exists : bool (default=False)
            If True, raise and exception if the repository
            already exists

        Returns
        -------
        RepoTree object representing the new repository
        """
        if not isidentifier(name):
            raise ValueError("Name must be a valid python identifier, got '%s'" % name)


        repo_root = os.path.join(self.idr_prop['repo_root'], name)

        if name in self.__sub_repo_table:
            if err_on_exists:
                raise ValueError("Repo %s already exists" % repo_root)
        else:
            self.add_repo_tree(RepoTree(repo_root=repo_root, parent_repo=self))

        #self._append_to_master_log(operation='mkrepo')

        return self.__sub_repo_table[name]

    def rmrepo(self):
        """
        Removes an empty repo

        Returns
        -------
            None
        """
        obj_cnt = len(self.__repo_object_table)
        rep_cnt = len(self.__sub_repo_table)
        if (obj_cnt > 0) or (rep_cnt > 0):
            msg = 'This repo has items in it (sub-repos or objects),'
            msg += " only emtpy repositories can be removed"
            raise ValueError(msg)

        os.rmdir(self.idr_prop['repo_root'])
        self.idr_prop['parent_repo'].refresh()

    def iterobjs(self, progress_bar=True):
        to_iter = self.iterleaves(progress_bar)
        for l in to_iter:
            yield l.load()

    def iterleaves(self, progress_bar=True):
        to_iter = sorted(self.__repo_object_table.keys())
        if progress_bar:
            try:
                from tqdm import tqdm
                with tqdm(total=len(self.__repo_object_table)) as pbar:
                    for k in to_iter:
                        pbar.set_description(k)
                        yield self.__repo_object_table[k]
                        pbar.update(1)
            except ImportError:
                for k in to_iter:
                    yield self.__repo_object_table[k]
        else:
            for k in to_iter:
                yield self.__repo_object_table[k]

    def iterrepos(self, progress_bar=True):
        to_iter = sorted(self.__sub_repo_table.keys())
        if progress_bar:
            try:
                from tqdm import tqdm
                with tqdm(total=len(self.__sub_repo_table)) as pbar:
                    for k in to_iter:
                        pbar.set_description(k)
                        yield self.__sub_repo_table[k]
                        pbar.update(1)
            except ImportError:
                for k in to_iter:
                    yield self.__sub_repo_table[k]
        else:
            for k in to_iter:
                yield self.__sub_repo_table[k]

    def list(self, list_repos=True, list_leaves=True):
        """
        List items inside a repository

        Parameters
        ----------
        list_repos : bool (default=True)
            Include repositories in the listing
        list_leaves : bool (default=True)
            Include objects in the listing
        verbose : bool (default=False)
            Unused

        Returns
        -------
        List of strings
        """

        objs = list(sorted(self.__repo_object_table.keys()))
        repos = list(sorted(self.__sub_repo_table.keys()))

        if list_repos and list_leaves:
            return repos, objs
        elif list_repos:
            return repos
        elif list_leaves:
            return objs
        else:
            raise ValueError("List repos and list objs set to False - nothing to do")

    def list_repos(self):
        return self.list(list_repos=True, list_leaves=False)

    def list_leaves(self):
        return self.list(list_repos=False, list_leaves=True)

    def from_reference(self, ref):
        if ref is None:
            msg = "Parameter 'ref' cannot be None"
            raise ValueError(msg)

        self_root = self.get_root()
        # If the reference is already a leaf
        if isinstance(ref, RepoLeaf):
            ref_root = ref.parent_repo.get_root()
            if self_root != ref_root:
                msg = "The ref object has a different root than this tree."
                msg += "\nTree root: %s ; Reference's root: %s" % (ref_root, self_root)
                raise ValueError(msg)
            return ref
        elif isinstance(ref, RepoTree):
            ref_root = ref.get_root()
            if self_root != ref_root:
                msg = "The ref object has a different root than this tree."
                msg += "\nTree root: %s ; Reference's root: %s" % (ref_root, self_root)
                raise ValueError(msg)
            return ref

        ref = ref.replace(URI_SPEC, '')
        nodes = ref.split('/')
        root_repo = self_root

        if root_repo.name != nodes[0]:
            msg = "Reference string is not absolute! Expected root '%s', got '%s'"
            msg = msg % (root_repo.name, nodes[0])
            raise ValueError(msg)

        curr_node = root_repo
        # Slice: First is root, last is type
        for n in nodes[1:]:
            try:
                if isinstance(curr_node, RepoLeaf):
                    break
                curr_node = curr_node[n]
            except KeyError as ke:
                return None

        return curr_node

    def summary(self):
        """
        Print basic plaintext summary of the repository
        """
        repos, objs = self.list()
        print("---Repos---")
        print("\n".join(repos))

        print("---Objects---")
        print("\n".join(objs))


#def reinit_metadata()

## ReInit-metadata option
## PARAMS: reinit_md, md_port=func
## - Set Metadata.metadata_port = md_port
##      - Default port should try to extract user_md in a few ways
##        but returns a new MD
## - Load all metadata into a map of uri to the md
##
## - Save this dict in root and