# Author: Morgan Stuart
import shutil
import abc
import string
import json
import pickle
import pandas as pd
import os
import time
from datetime import datetime
from glob import glob
from ast import parse
import sys


IS_PYTHON3 = sys.version_info > (3, 0)
try:
  basestring
except NameError:
  basestring = str

if IS_PYTHON3:
    fs_except = FileExistsError
    prompt_input = input
else:
    prompt_input = raw_input
    fs_except = OSError


def isidentifier(name):
    try:
        parse('{} = None'.format(name))
        return True
    except (SyntaxError, ValueError, TypeError) as e:
        return False

URI_SPEC = 'datatree://'
idr_config = dict(storage_root_dir=os.path.join(os.path.expanduser("~"), '.idt_root'),
                  master_log='LOG', master_index='INDEX',
                  repo_extension='repo', metadata_extension='mdjson',
                  lock_extension='lock',
                  date_format='%h %d %Y (%I:%M:%S %p)',
                  enable_queries=True)

reserved_md_keywords = {'write_time', 'extra_metadata_keys',
                        'obj_type', 'referrers', 'references'}

shared_metadata = dict(notebook_name=None, author=None, project=None)
base_master_log_entry = dict(repo_tree=None,
                             repo_leaf=None,
                             storage_type=None,
                             repo_operation=None,
                             timestamp=None,
                             notebook_path=None, author=None)

# - monkey patching docstrings
# - add a log of all operations in root
#   - Most recent (list most recently accessed/written)

# - Repo (dir) metadata - store create time of repo
# - Better support for dot paths (relative repo provided on save)
# - Enable references between objects
#   - Simply store a set of dot paths in the metadata
# - Log interface for easier adjustable verbosity and logging

def is_valid_uri(obj):
    if not isinstance(obj, basestring):
        return False
    elif obj[:len(URI_SPEC)] == URI_SPEC:
        return True
    else:
        return False


def message_user(txt):
    print(txt)

# idt.set_local_var_to_notebook_name(var_name="""idt.shared_metadata[\\'notebook_name\\']""")
def set_local_var_to_notebook_name(var_name='NOTEBOOK_NAME'):
    from IPython import get_ipython
    ipython = get_ipython()

    js = """IPython.notebook.kernel.execute('%s = ' + '"' + IPython.notebook.notebook_name + '"')""" % var_name
    ipython.run_cell_magic('javascript', '', js)

def norm_vector(v):
    return sum(_v**2.0 for _v in v)

# Same as dot
def inner_prod_vector(a, b):
    return sum(_a * _b for _a, _b in zip(a, b))

def cosine_sim(a, b):
    na = norm_vector(a)
    nb = norm_vector(b)
    sim = inner_prod_vector(a, b)/(na * nb)
    return sim

def clean_token(tk):
    remove_chars = set(""",.!@#$%^&*()[]{}/\\`~-+|;:' \t\n\r""")
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
class LockFile(object):
    """
    A context object (use in 'with' statement) that attempts to create
    a lock file on entry, with blocking/retrying until successful
    """
    def __init__(self, path, poll_interval=1,
                 wait_msg=None):
        """
        Parameters
        ----------
        path : lock file path as string
        poll_interval : integer
            Time between retries while blocking on lock file
        wait_msg : str
            Message to print to user if this lock starts blocking
        """
        self.path = path
        self.poll_interval = poll_interval
        self.wait_msg = wait_msg
        self.locked = False

        # Make sure the directory exists
        dirs = os.path.split(path)[:-1]
        p = os.path.join(*dirs)
        valid_path = os.path.isdir(p)
        if not valid_path:
            msg = "The lock path '%s' is not since '%s' is not a directory"
            raise ValueError(msg % (path, p))

    def __enter__(self):
        block_count = 0
        while True:
            try:
                self.fs_lock = os.open(self.path,
                                       os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except fs_except as e:
                block_count += 1
                if self.wait_msg is not None:
                    print("[%d] %s" % (block_count, self.wait_msg))
                time.sleep(self.poll_interval)
        self.locked = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(self.fs_lock)
        os.remove(self.path)

def leaf_to_reference(obj, to_storage_interface=False):
    if isinstance(obj, basestring):
        return obj
    elif isinstance(obj, RepoLeaf):
        if to_storage_interface:
            return obj._get_highest_priority_si().reference()
        else:
            return obj.reference()
    elif isinstance(obj, StorageInterface):
        return obj.reference()
    else:
        return obj

def reference_to_leaf(tree, obj):
    if isinstance(obj, basestring) and is_valid_uri(obj):
        return tree.from_reference(obj)
    elif isinstance(obj, basestring):
        return obj
    else:
        return obj


standard_metadata = ['author', 'comments', 'tags',
                     'write_time', 'obj_type']
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
        self.parent_leaf = parent_leaf
        self.name = parent_leaf.name
        self.path = parent_leaf.save_path + '.' + self.extension

        self.lock_file = self.path + '.' + idr_config['lock_extension']

        self.md_path = self.path + '.' + idr_config['metadata_extension']
        self.lock_md_file = self.md_path + '.' + idr_config['lock_extension']
        self.init()

    def __call__(self, *args, **kwargs):
        return self.load()

    def __str__(self):
        #return self.parent_leaf.reference(storage_type=self.storage_name)
        repos = self.parent_leaf.parent_repo.get_parent_repo_names()
        repos[0] = 'ROOT'
        return ".".join(repos + [self.parent_leaf.name, self.storage_name])

    def get_associated_files_and_locks(self):
        return [(self.path, self.lock_file),
                (self.md_path, self.lock_md_file)]

    @classmethod
    def get_missing_metadata_fields(cls, md):
        if not isinstance(md, dict):
            raise ValueError("Expected dict, got %s" % str(type(md)))

        if StorageInterface.required_metadata is None:
            req_md = list()
        elif not isinstance(StorageInterface.required_metadata, list):
            req_md = [cls.required_metadata]
        else:
            req_md = cls.required_metadata

        missing = list()
        for rm in req_md:
            if rm not in md:
                missing.append(rm)
        return missing

    @staticmethod
    def __collapse_metadata_deltas(md_entries):
        if not isinstance(md_entries, list):
            msg = "Expected md_entries to be a list, got %s instead"
            raise ValueError(msg % str(type(md_entries)))

        if not all(isinstance(d, dict) for d in md_entries):
            msg = "Entries in metadata must be dict"
            raise ValueError(msg)

        # TODO: track first/last time a key was updated in MD
        latest_md = dict()
        for md_d in md_entries:
            latest_md.update(md_d)
        return latest_md

    def init(self):
        pass

    def exists(self):
        return os.path.isfile(self.path)

    def md_exists(self):
        return os.path.isfile(self.md_path)

    def load(self):
        """
        Locks the object and reads the data from the filesystem

        Returns
        -------
        Object stored
        """
        with LockFile(self.lock_file):
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
        missing_md = StorageInterface.get_missing_metadata_fields(md_kwargs)
        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        with LockFile(self.lock_file):
            with open(self.path, mode='wb') as f:
                pickle.dump(obj, f)

            self.write_metadata(obj=obj, **md_kwargs)

    def reference(self):
        r_names = self.parent_leaf.parent_repo.get_parent_repo_names()
        type_ext = self.storage_name
        r_names.append(self.parent_leaf.parent_repo.name)
        r_names.append(self.parent_leaf.name)
        r_names.append(type_ext)
        ref_str = '/'.join(r_names)
        ref_str = URI_SPEC + ref_str
        return ref_str

    def add_reference(self, reference_uri, *reference_md_keys):
        md_hist = self.read_metadata(lock=True,
                                     most_recent=False,
                                     resolve_references=False)
        last_md = md_hist[-1]
        references = last_md.get('references', dict())
        keys = references.get(reference_uri, list())

        references[reference_uri] = list(set(keys + list(reference_md_keys)))
        for k in references[reference_uri]:
            last_md[k] = reference_uri

        last_md['references'] = references
        md_hist[-1] = last_md

        with LockFile(self.lock_md_file):
            with open(self.md_path, 'w')  as f:
                json.dump(md_hist, f)

    def remove_reference(self, reference_uri, *reference_md_keys):
        md_hist = self.read_metadata(lock=True,
                                     most_recent=False,
                                     resolve_references=False)
        last_md = md_hist[-1]
        references = last_md.get('references', dict())

        if reference_uri not in references:
            msg = "URI '%s' not in '%s' references" % (reference_uri, self.name)
            raise ValueError(msg)

        keys = references[reference_uri]
        for k in keys:
            last_md[k] = '[DELETED]' + str(last_md[k])

        del references[reference_uri]
        last_md['references'] = references
        md_hist[-1] = last_md

        with LockFile(self.lock_md_file):
            with open(self.md_path, 'w')  as f:
                json.dump(md_hist, f)


    def add_referrer(self, referrer_uri, *referrer_md_keys):
        md_hist = self.read_metadata(lock=True,
                                     most_recent=False,
                                     resolve_references=False)
        last_md = md_hist[-1]

        referrers = last_md.get('referrers', dict())

        if referrer_uri not in referrers:
            referrers[referrer_uri] = list()

        referrers[referrer_uri].extend(referrer_md_keys)
        referrers[referrer_uri] = list(set(referrers[referrer_uri]))
        last_md['referrers'] = referrers
        md_hist[-1] = last_md

        with LockFile(self.lock_md_file):
            with open(self.md_path, 'w')  as f:
                json.dump(md_hist, f)

    def remove_referrer(self, referrer_uri, *referrer_md_keys):
        md_hist = self.read_metadata(lock=True,
                                     most_recent=False,
                                     resolve_references=False)
        last_md = md_hist[-1]
        if 'referrers' not in last_md:
            raise ValueError("Cannot remove a referrer that doesn't exist")

        referrers = last_md['referrers']

        if referrer_uri not in referrers:
            raise ValueError("URI '%s' is not a referrer to %s" %
                             (referrer_uri, self.name))

        for k in referrer_md_keys:
            referrers[referrer_uri].remove(k)

        if len(referrers[referrer_uri]) == 0:
            del referrers[referrer_uri]

        last_md['referrers'] = referrers
        md_hist[-1] = last_md

        with LockFile(self.lock_md_file):
            with open(self.md_path, 'w')  as f:
                json.dump(md_hist, f)

    # update references, not referrers
    # - Referrers lets us track back to the referencers, who we need to change
    #   their references if the referred is being moved
    def update_referrer(self, orig_referrer_uri, new_referrer_uri,
                        orig_referrer_md_key, new_referrer_md_key):
       self.remove_referrer(orig_referrer_uri, orig_referrer_md_key)
       self.add_referrer(new_referrer_uri, new_referrer_md_key)

    def write_metadata(self, obj=None, **md_kwargs):
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

        references = dict()
        for k in md_kwargs.keys():
            # Overwrite leaf/URI to just URI
            md_kwargs[k] = leaf_to_reference(md_kwargs[k],
                                             to_storage_interface=True)

            # Normalize back to a leaf so we can update it's MD
            # store
            if is_valid_uri(md_kwargs[k]):
                uri = md_kwargs[k]
                if uri not in references:
                    references[uri] = list()

                references[uri].append(k)

        if obj is not None:
            md_kwargs['obj_type'] = type(obj).__name__

        cls = self.__class__
        md_kwargs['write_time'] = datetime.now().strftime(idr_config['date_format'])
        md_kwargs['extra_metadata_keys'] = list(set(md_kwargs.keys()) -
                                                 set(cls.required_metadata +
                                                     standard_metadata +
                                                     cls.interface_metadata))
        md_kwargs['references'] = references

        with LockFile(self.lock_md_file):
            md = self.read_metadata(lock=False, most_recent=False,
                                    resolve_references=False)
            most_recent_md = StorageInterface.__collapse_metadata_deltas(md)

            # TODO?
            # Remove this leaf from referrers to other
            # leafs if a reference was removed
            #current_refs = most_recent_md.get('references', dict())
            this_uri = self.reference()
            for uri, keys in references.items():
                l = reference_to_leaf(self.parent_leaf.parent_repo, uri)

                if isinstance(l, StorageInterface):
                    l.add_referrer(this_uri, *keys)
                elif isinstance(l, RepoLeaf):
                    # No type was specified, so reference the highest priority
                    si = l._get_highest_priority_si()
                    si.add_referrer(this_uri, *keys)
                elif isinstance(l, RepoTree):
                    msg = ("Cannot reference a repository (%s), found in md keys: "
                           % l.name)
                    msg += ", ".join(keys)
                    raise ValueError(msg)



            # Remove key values that we already have stored (key AND value match)
            for k, v in most_recent_md.items():
                if k in md_kwargs and v == md_kwargs[k]:
                    del md_kwargs[k]

            md.append(md_kwargs)
            with open(self.md_path, 'w') as f:
                json.dump(md, f)

    def read_metadata(self, lock=True, most_recent=True,
                      resolve_references=True):
        """
        Read entire metadata history from storage, with optional
        locking.

        Parameters
        ----------
        lock : bool (default=True)
            Whether or not to lock the metadata file before reading.

        Returns
        -------
        Metadata history as a list of dictionaries
        """
        if os.path.isfile(self.md_path):
            if lock:
                with LockFile(self.lock_md_file):
                    with open(self.md_path, 'r') as f:
                        md = json.load(f)
            else:
                with open(self.md_path, 'r') as f:
                    md = json.load(f)
        else:
            md = []

        if most_recent:
            md = StorageInterface.__collapse_metadata_deltas(md)

        if resolve_references:
            if most_recent:
                for k in md.keys():
                    md[k] = reference_to_leaf(self.parent_leaf.parent_repo, md[k])
            else:
                for _md in md:
                    for k in _md.keys():
                        _md[k] = reference_to_leaf(self.parent_leaf.parent_repo,
                                                   _md[k])

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
        md = self.read_metadata()
        str_md_terms = [basic_tokenizer(v) for k, v in md.items()
                        if isinstance(v, basestring)]
        str_md_terms = [_v for v in str_md_terms for _v in v]
        return str_md_terms

    @staticmethod
    def _build_html_body_(md):

        html_str = """
        <b>Author</b>: {author} <br>
        <b>Last Write</b>: {ts} <br>
        <b>Comments</b>: {comments} <br>
        <b>Type</b>: {ty} <br>
        <b>Tags</b>: {tags} <br>
        """.format(author=md.get('author'),
                   comments=md.get('comments'), ts=md.get('write_time'),
                   ty=md.get('obj_type'), tags=md.get('tags'))


        extra_keys = md.get('extra_metadata_keys', list())
        if len(extra_keys) > 0:
            html_items = ["<b>%s</b>: %s <br>" % (k, str(md[k])[:50])
                          for k in extra_keys]
            add_md = """
            <h4>Additional Metadata</h4>
            %s
            """ % "\n".join(html_items)
            html_str += add_md

        return html_str

    def _repr_html_(self):
        md = self.read_metadata()
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
        with LockFile(self.lock_file):
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

        with LockFile(self.lock_file):
            hdf_store = pd.HDFStore(self.path, mode='w')
            hdf_store.put(HDFStorageInterface.hdf_data_level,
                          obj, format=HDFStorageInterface.hdf_format)
            hdf_store.close()

        self.write_metadata(obj=obj, **md_kwargs)

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
        with LockFile(self.lock_file):
            obj = pd.read_hdf(self.path, mode='r', stop=n)
        return obj

    def write_metadata(self, obj=None, **md_kwargs):
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

        if isinstance(obj, pd.DataFrame):
            md_kwargs['columns'] = list(str(c) for c in obj.columns)
            md_kwargs['dtypes'] = list(str(d) for d in obj.dtypes)

        if obj is not None:
            md_kwargs['index_head'] = list(str(i) for i in obj.index[:5])
            md_kwargs['length'] = len(obj)

        super(HDFStorageInterface, self).write_metadata(obj=obj, **md_kwargs)

    def _repr_html_(self):
        md = self.read_metadata()

        basic_descrip = StorageInterface._build_html_body_(md)
        extra_descrip = """
        <b>Num Entries </b> : {num_entries} <br>
        <b>Columns</b> ({n_cols}) : {col_sample} <br>
        <b>Index Head</b> : {ix_head} <br>
        """.format(num_entries=md.get('length'),
                   n_cols=len(md.get('columns')),
                   col_sample=", ".join(md.get('columns', [])[:10]),
                   ix_head=", ".join(md.get('index_head', [])))


        div_template = """
        <div style="width: 80%;">
            <div style="float:left; width: 40%">
            {basic_description}
            </div>
            <div style="float:right;">
            {extra_description}
            </div>
        </div>
        """.format(basic_description=basic_descrip,
                   extra_description=extra_descrip)

        html_str = """<h2> {name} </h2>""".format(name=self.name)
        html_str += div_template
        return html_str


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

        missing_md = StorageInterface.get_missing_metadata_fields(md_kwargs)
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

        return style_html + pygment_html

class SQLStorageInterface(StorageInterface):
    storage_name = 'sql'
    extension = 'sql'
    expose_on_leaf = ['query']
    required_metadata = []

    def load(self):
        with LockFile(self.lock_file):
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

        missing_md = self.get_missing_metadata_fields(md_kwargs)

        if len(missing_md) > 0:
            msg = "Missing required metadata fields: %s"
            raise ValueError(msg % ", ".join(missing_md))

        store_dict = dict(obj.asdict())
        with LockFile(self.lock_file):
            with open(self.path, mode='w') as f:
                json.dump(store_dict, f)

            self.write_metadata(obj=obj, **md_kwargs)

    def query(self, cxn, **parameters):
        if not hasattr(cxn, 'query'):
            msg = 'Argument cxn must have a query method'
            raise ValueError(msg)

        q = self.load().build_query(**parameters)
        #q = SQLStorageInterface.build_query(self.load())
        return cxn.query(q)

    def _repr_html_(self):
        md = self.read_metadata()

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

        div_template = """
        <div style="width: 80%;">
            <div style="float:left; width: 40%">
                {basic_description}
            </div>
            <div style="float:right;">
                {sql_txt}
            </div>
        </div>
        """.format(basic_description=basic_descrip,
                   sql_txt=style_html + pygment_html)

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
        msg = "%s has specified the extension '%s', which is alread in use by %s"
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


register_storage_interface(HDFStorageInterface, 'hdf', 0,
                           types=[pd.DataFrame, pd.Series])
register_storage_interface(StorageInterface, 'pickle', 1)
register_storage_interface(ModelStorageInterface, 'model', 2)
register_storage_interface(SQLStorageInterface, 'sql', 2,
                           types=[SQL])


class RepoLeaf(object):
    """
    A terminating node in the repo tree, providing an interface
    to possibly several objects saved using different storage types.
    """
    # ABC used in order to allow doc string monkey patching
    __metaclass__ = abc.ABCMeta

    def __init__(self, parent_repo, name):
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
        self.refresh()

    def __call__(self, *args, **kwargs):
        """
        Load the object stored under this name, uses
        storage interface priority list.

        Returns
        -------
        Stored data object

        """
        return self.load(storage_type=None)

    def __getitem__(self, item):
        if item not in self.type_to_storage_interface_map:
            msg = "No storage interface '%s'. Expected one of %s"
            msg = msg % (item, ",".join(self.type_to_storage_interface_map.keys()))
            raise KeyError(msg)

        return self.type_to_storage_interface_map[item]

    def __update_doc_str(self):
        docs = self.name + "\n\n"
        si = self._get_highest_priority_si()

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

    def _get_highest_priority_si(self):
        for st in storage_type_priority_order:
            if st in self.type_to_storage_interface_map:
                return self.type_to_storage_interface_map[st]

    def _repr_html_(self):
        return self._get_highest_priority_si()._repr_html_()

    def get_vector_representation_map(self):
        self.refresh()
        return {ty:si.get_vector_representation()
                for ty, si in self.type_to_storage_interface_map.items()}

    def reference(self, storage_type=None):
        r_names = self.parent_repo.get_parent_repo_names()

        if storage_type is None:
            type_ext = self._get_highest_priority_si().storage_name
        else:
            type_ext = self.type_to_storage_interface_map[storage_type].storage_name

        r_names.append(self.parent_repo.name)
        r_names.append(self.name)
        r_names.append(type_ext)

        ref_str = '/'.join(r_names)
        ref_str = URI_SPEC + ref_str
        return ref_str

    def refresh(self):
        mde = idr_config['metadata_extension']
        repe = idr_config['repo_extension']

        cur_si_map = dict(self.type_to_storage_interface_map)
        cur_types = set(cur_si_map.keys())

        # Map extenstions of files to their paths
        self.tmp = {os.path.split(p)[-1].split('.')[-1]: p
                                              for p in glob(self.save_path + '*')
                                              if p[-len(mde):] != mde
                                              and p[-len(repe):] != repe
                                              }
        # Create maps of SI to paths
        self.tmp = {extension_to_interface_name_map[k]: v
                    for k, v in self.tmp.items()}

        # Create new SI or grab current one
        self.type_to_storage_interface_map = {k: storage_interfaces[k](parent_leaf=self)
                                              if k not in self.type_to_storage_interface_map else
                                              self.type_to_storage_interface_map[k]
                                              for k, v in self.tmp.items()}

        # Delete types that are no longer present on the FS
        next_types = set(self.type_to_storage_interface_map.keys())

        for t in (cur_types - next_types):
            si = cur_si_map[t]
            delattr(self, si.storage_name)

            for eol in si.expose_on_leaf:
                if hasattr(si, eol):#
                    if hasattr(self, eol) and getattr(self, eol) == getattr(si, eol):
                       delattr(self, eol)

        if len(next_types) > 0:
            for t in next_types:
                si = self.type_to_storage_interface_map[t]
                setattr(self, si.storage_name, si)

            # Only expose SI attrs from the highest priority SI
            next_types = sorted(list(next_types),
                                key=lambda t: storage_type_priority_map[t])
            priority_type = self.type_to_storage_interface_map[next_types[0]]

            for eol in priority_type.expose_on_leaf:
                if hasattr(si, eol):
                    setattr(self, eol, getattr(si, eol))

        self.__update_doc_str()
        return self

    def read_metadata(self, storage_type=None, **kwargs):
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
        if storage_type is not None:
            if storage_type not in self.type_to_storage_interface_map:
                raise ValueError("Type %s does not exist for %s" % (storage_type, self.name))

            md = self.type_to_storage_interface_map[storage_type].read_metadata(**kwargs)
        else:
            md = None
            for po in storage_type_priority_order:
                if po in self.type_to_storage_interface_map:
                    md = self.type_to_storage_interface_map[po].read_metadata(**kwargs)
                    break
        return md

    def save(self, obj, storage_type=None, auto_overwrite=False,
             verbose=True,
             **md_props):
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
            storage_type = type_storage_lookup.get(type(obj), 'pickle')
        elif storage_type not in storage_interfaces:
            msg = "Unknown storage type '%s', expecting one of %s"
            msg = msg % (storage_type, ",".join(storage_interfaces.keys()))
            raise ValueError(msg)

        # Construct the filesystem path using a typed extension
        store_int = storage_interfaces[storage_type](parent_leaf=self)

        # Double Check - file exists there or file is registered in memory
        if store_int.exists() or storage_type in self.type_to_storage_interface_map:
            if auto_overwrite and verbose:
                print("Auto overwriting '%s' (%s) in %s " % (self.name,
                                                             storage_type,
                                                             self.parent_repo.name))
            elif not auto_overwrite:
                prompt = "An object named '%s' (%s) in %s already exists" % (self.name,
                                                                             storage_type,
                                                                             self.parent_repo.name)
                prompt += "\nOverwrite this object? (y/n)"
                y = prompt_input(prompt)
                if y == 'y' or y == 'yes':
                    print("Proceeding with overwrite...")
                else:
                    print("Aborting...")
                    return

        if verbose:
            message_user("Saving to: %s.%s (%s)" % (self.parent_repo.name,
                                                    self.name, storage_type))

        store_int.save(obj, **md_props)

        if verbose:
            message_user("Save Complete")
            message_user("-------------")

        self.parent_repo._append_to_master_log(operation='save', leaf=self,
                                               author=md_props.get('author', None),
                                               storage_type=storage_type)

        self.parent_repo._add_to_index(leaf=self,
                                       storage_type=storage_type)

        self.refresh()

    # TODO: Move delete to storage interface, perform with lock?
    def delete(self, author, storage_type=None):
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
        orig_si_uris = {ty:si.reference()
                        for ty, si in self.type_to_storage_interface_map.items()}

        if storage_type is None:
            filenames = glob(self.save_path + '.*')
            message_user("Deleting:\n%s" % "\n".join(filenames))
            [os.remove(fn) for fn in filenames]
        else:
            p = self.type_to_storage_interface_map[storage_type].path
            md_p = self.type_to_storage_interface_map[storage_type].md_path
            os.remove(p)
            os.remove(md_p)

        self.parent_repo._append_to_master_log(operation='delete', leaf=self,
                                               author=author,
                                               storage_type=storage_type)
        self.parent_repo._remove_from_index(self, storage_type=storage_type, missing_err='ignore')
        self.parent_repo.refresh()

    def rename(self, new_name, author):
        if not isidentifier(new_name):
            msg = "New name must be a valid python identifier, got '%s'" % new_name
            raise ValueError(msg)

        if new_name in self.parent_repo.list(list_repos=False):
            msg = ("Repo '%s' already has a leaf with name '%s'"
                   % (self.parent_repo.name, new_name))
            raise ValueError(msg)

        orig_si_uris = {ty:si.reference()
                        for ty, si in self.type_to_storage_interface_map.items()}

        orig_name = self.name
        base_p = self.parent_repo.idr_prop['repo_root']
        for ty, si in self.type_to_storage_interface_map.items():
            files_and_locks = si.get_associated_files_and_locks()
            # Remove all files using associated lock files
            for f, l_f in files_and_locks:
                # Will break if lock file is not valid...
                with LockFile(l_f):
                    fname = os.path.split(f)[-1]
                    new_fname = fname.replace(orig_name, new_name)
                    new_p = os.path.join(base_p, new_fname)
                    shutil.move(f, new_p)

            # Remove storage type from the index
            self.parent_repo._remove_from_index(self,
                                                storage_type=ty)

        self.parent_repo.refresh()

        self.parent_repo._append_to_master_log(operation='rename', leaf=self,
                                               author=author,
                                               storage_type=None)

        self.parent_repo._add_to_index(self, storage_type=None)

        self.parent_repo[new_name].update_references(orig_si_uris)

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
        orig_si_uris = {ty:si.reference()
                        for ty, si in self.type_to_storage_interface_map.items()}

        for ty, si in self.type_to_storage_interface_map.items():
            files_and_locks = si.get_associated_files_and_locks()
            # Remove all files using associated lock files
            for f, l_f in files_and_locks:
                # Will break if lock file is not valid...
                with LockFile(l_f):
                    fname = os.path.split(f)[-1]
                    new_p = os.path.join(new_base_p, fname)
                    shutil.move(f, new_p)

            # Remove storage type from the index
            self.parent_repo._remove_from_index(self,
                                                storage_type=ty)
        self.parent_repo.refresh()
        tree.refresh()
        self.parent_repo._append_to_master_log(operation='move', leaf=self,
                                               author=author,
                                               storage_type=None)

        self.parent_repo._add_to_index(tree[self.name], storage_type=None)

        tree[self.name].update_references(orig_si_uris)

    def load(self, storage_type=None):
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
        store_int = None
        if storage_type is None:
            for po in storage_type_priority_order:
                if po in self.type_to_storage_interface_map:
                    store_int = self.type_to_storage_interface_map[po]
                    break
        else:
            store_int = self.type_to_storage_interface_map[storage_type]


        return store_int.load()

    def update_references(self, previous_si_uris, delete=False):
        # Go through new interfaces, make sure that any interfaces that
        # were referring to this interface before move are updated
        for ty, si in self.type_to_storage_interface_map.items():
            md = si.read_metadata(resolve_references=False)
            # If self has been deleted, we only need to remove references/referrers
            # so we shouldn't try and get the current URI since it is irrelevant
            if not delete:
                new_si_uri = si.reference()

            referrers = md.get('referrers', dict())
            for referrer_uri, keys in referrers.items():
                # 'l' is a leaf that references this leaf
                # -> Since this leaf is moving, we must update l's references
                l = reference_to_leaf(self.parent_repo, referrer_uri)
                assert isinstance(l, StorageInterface)
                l.remove_reference(previous_si_uris[ty], *keys)
                if not delete:
                    l.add_reference(new_si_uri, *keys)

            references = md.get('references', dict())
            for reference_uri, keys in references.items():
                # 'l' is a leaf that this leaf references
                # -> Since this leaf is moving, we must update l's referrers
                l = reference_to_leaf(self.parent_repo, reference_uri)
                l.remove_referrer(referrer_uri=previous_si_uris[ty], *keys)
                if not delete:
                    l.add_referrer(referrer_uri=new_si_uri, *keys)


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
        self.refresh()

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
        """ % "\n".join("<li>%s</li>" % rt for rt in sorted(self.__sub_repo_table.keys()))
        objects_html = """
        <h4>Objects</h4>
        %s
        """ % "\n".join("<li>%s</li>" % rt for rt in sorted(self.__repo_object_table.keys()))

        if self.idr_prop['parent_repo'] is not None:
            parent_repo_str = "->".join(['Root']
                                        + self.get_parent_repo_names()[1:]
                                        + [self.name])
            #parent_repo_str += "-> %s" % self.name
        else:
            parent_repo_str = "Root (%s)" % self.name
        html = """
        {repo_parent_header}
        <div style="width: 30%;">
            <div style="float:left; width: 50%">
            {repos_list}
            </div>
            <div style="float:right;">
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
            root_repo.save(dict(), name=ix_name, author='system',
                           comments='index of objects across entire tree',
                           tags='idt_index', verbose=False)
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

    def _remove_from_index(self, leaf, storage_type, missing_err='raise'):
        if missing_err not in ('ignore', 'raise'):
            raise ValueError("Unknown value for param missing_err, expected one of 'raise', 'ignore'")

        master_index = self._load_master_index()
        if storage_type is not None:
            ref = leaf.reference(storage_type=storage_type)
            if ref in master_index:
                del master_index[ref]
            elif missing_err == 'raise':
                raise ValueError("'%s' is not in index" % ref)
        else:
            for st in leaf.type_to_storage_interface_map.keys():
                ref = leaf.reference(storage_type=st)
                if ref in master_index:
                    del master_index[ref]
                elif missing_err == 'raise':
                    raise ValueError("'%s' is not in index" % ref)

        self._write_master_index(master_index)

    def _add_to_index(self, leaf, storage_type):
        # Avoid indexing the index
        if leaf.name in (idr_config['master_log'], idr_config['master_index']):
            return

        # Open index object in root tree
        #   - Each type has its own index, so use storage type to distinguish
        vec_map = leaf.get_vector_representation_map()
        master_index = self._load_master_index()
        if storage_type is not None:
            ref = leaf.reference(storage_type=storage_type)
            master_index[ref] = vec_map[storage_type]
        else:
            for st, vec in vec_map.items():
                ref = leaf.reference(storage_type=st)
                master_index[ref] = vec

        self._write_master_index(master_index)

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

    def refresh(self):
        """
        Force the tree to rebuild by rescanning the filesystem

        Returns
        -------
        self
        """
        repo_curr = dict(self.__sub_repo_table)
        obj_curr = dict(self.__repo_object_table)
        self.__sub_repo_table = dict()
        self.__repo_object_table = dict()

        all_dir_items = os.listdir(self.idr_prop['repo_root'])
        matching_items = [f for f in all_dir_items
                          if '.' in f
                          and f[-len(idr_config['metadata_extension']):] != idr_config['metadata_extension']]

        for f in matching_items:
            base_name, ext = f.split('.')
            is_file = os.path.isfile(os.path.join(self.idr_prop['repo_root'], f))
            if is_file:
                # Had the object before - don't create new Leaf
                if base_name in obj_curr:
                    self.add_obj_leaf(obj_curr[base_name].refresh())
                else:
                    self.add_obj_leaf(RepoLeaf(parent_repo=self, name=base_name))
            else:
                if base_name in repo_curr:
                    self.add_repo_tree(repo_curr[base_name].refresh())
                else:
                    p = os.path.join(self.idr_prop['repo_root'], f)
                    self.add_repo_tree(RepoTree(repo_root=p, parent_repo=self))


        curr_names = set(list(obj_curr.keys()) + list(repo_curr.keys()))
        new_names = set(list(self.__repo_object_table.keys()) + list(self.__sub_repo_table.keys()))
        for n in curr_names - new_names:
            delattr(self, n)

        return self

    def delete(self, name, author, storage_type=None):
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
        self.__repo_object_table[name].delete(author=author, storage_type=storage_type)
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

        leaf = self.__repo_object_table.get(name, RepoLeaf(parent_repo=self, name=name))
        leaf.save(obj, storage_type=storage_type,
                  auto_overwrite=auto_overwrite,
                  **extra_kwargs)
        #leaf.save(obj, storage_type=storage_type, author=author,
        #          auto_overwrite=auto_overwrite, comments=comments, tags=tags,
        #          **extra_kwargs)

        self.add_obj_leaf(leaf)

    def load(self, name, storage_type=None):
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
        if storage_type is not None:
            st = st[storage_type]
        return st.load()

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

        self._append_to_master_log(operation='mkrepo')

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

    def iterobjs(self):
        #for k in sorted(self.__repo_object_table.keys()):
        #    yield self.__repo_object_table[k].load()
        for l in self.iterleaves():
            yield l.load()

    def iterleaves(self):
        for k in sorted(self.__repo_object_table.keys()):
            yield self.__repo_object_table[k]

    def iterrepos(self):
        for k in sorted(self.__sub_repo_table.keys()):
            yield self.__sub_repo_table[k]

    def list(self, list_repos=True, list_objs=True, verbose=False):
        """
        List items inside a repository

        Parameters
        ----------
        list_repos : bool (default=True)
            Include repositories in the listing
        list_objs : bool (default=True)
            Include objects in the listing
        verbose : bool (default=False)
            Unused

        Returns
        -------
        List of strings
        """

        objs = list(sorted(self.__repo_object_table.keys()))
        repos = list(sorted(self.__sub_repo_table.keys()))

        if list_repos and list_objs:
            return repos, objs
        elif list_repos:
            return repos
        elif list_objs:
            return objs
        else:
            raise ValueError("List repos and list objs set to False - nothing to do")

    def from_reference(self, ref):
        # If the reference is already a leaf
        if isinstance(ref, RepoLeaf):
            ref = ref._get_highest_priority_si()

        # Reference is already a storage interface
        if isinstance(ref, StorageInterface):
            ref_root = ref.parent_leaf.parent_repo.get_root()
            this_root = self.get_root()
            if ref_root != this_root:
                msg = "The ref object has a different root than this tree."
                msg += "\nTree root: %s ; Reference's root: %s" % (ref_root, this_root)
                raise ValueError(msg)
            else:
                return ref

        ref = ref.replace(URI_SPEC, '')
        nodes = ref.split('/')
        root_repo = self.get_root()

        if root_repo.name != nodes[0]:
            msg = "Reference string is not absolute! Expected root '%s', got '%s'"
            msg = msg % (root_repo.name, nodes[0])
            raise ValueError(msg)

        curr_node = root_repo
        # Slice: First is root, last is type
        for n in nodes[1:]:
            try:
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
