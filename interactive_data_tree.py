# Author: Morgan Stuart
import abc
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


idr_config = dict(storage_root_dir=os.path.join(os.path.expanduser("~"), '.idt_root'),
                  master_log='LOG',
                  repo_extension='repo', metadata_extension='mdjson',
                  lock_extension='lock',
                  date_format='%h %d %Y (%I:%M:%S %p)')

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

# idt.set_local_var_to_notebook_name(var_name="""idt.shared_metadata[\\'notebook_name\\']""")
def set_local_var_to_notebook_name(var_name='NOTEBOOK_NAME'):
    from IPython import get_ipython
    ipython = get_ipython()

    js = """IPython.notebook.kernel.execute('%s = ' + '"' + IPython.notebook.notebook_name + '"')""" % var_name
    ipython.run_cell_magic('javascript', '', js)

class LockFile(object):
    """
    A context object (use in 'with' statement) that attempts to create
    a lock file on entry, with blocking/retrying until successful
    """
    def __init__(self, path, poll_interval=1):
        """
        Parameters
        ----------
        path : lock file path as string
        poll_interval : integer
            Time between retries while blocking on lock file
        """
        self.path = path
        self.poll_interval = poll_interval
        self.locked = False

    def __enter__(self):
        while True:
            try:
                self.fs_lock = os.open(self.path,
                                       os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except fs_except as e:
                time.sleep(self.poll_interval)
        self.locked = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(self.fs_lock)
        os.remove(self.path)

#######
# Storage Interfaces
class StorageInterface(object):
    """
    Base storage interface representing an arbitrary python object
    stored using pickle. New storage interfaces should be derived
    from this class.
    """
    extension = 'pkl'

    def __init__(self, path, name):
        self.name = name
        if path[-len(self.extension):] != self.extension:
            self.path = path + '.' + self.extension
        else:
            self.path = path

        self.lock_file = self.path + '.' + idr_config['lock_extension']

        self.md_path = self.path + '.' + idr_config['metadata_extension']
        self.lock_md_file = self.md_path + '.' + idr_config['lock_extension']

    def exists(self):
        return os.path.isfile(self.path)

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
        with LockFile(self.lock_file):
            with open(self.path, mode='wb') as f:
                pickle.dump(obj, f)

            self.write_metadata(obj=obj, **md_kwargs)

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
        if obj is not None:
            md_kwargs['obj_type'] = type(obj).__name__

        md_kwargs['write_time'] = datetime.now().strftime(idr_config['date_format'])

        with LockFile(self.lock_md_file):
            md = self.read_metadata(lock=False)
            md.append(md_kwargs)
            with open(self.md_path, 'w') as f:
                json.dump(md, f)

    def read_metadata(self, lock=True):
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
        return md

    def get_vector_representation(self):
        termset = ['str', 'dataframe', 'series', 'query', 'data']
        term_cnts = {n:0 for n in termset}
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
        md = self.read_metadata()[-1]
        str_md_terms = [v for k, v in md.items() if isinstance(v, basestring)]
        return str_md_terms

    @staticmethod
    def _build_html_body_(md):

        html_str = """
        <b>Author</b>: {author} <br>
        <b>Last Write</b>: {ts} <br>
        <b>Comments</b>: {comments} <br>
        <b>Type</b>: {ty} <br>
        <b>Tags</b>: {tags} <br>
        """.format(author=md['author'],
                   comments=md['comments'], ts=md['write_time'],
                   ty=md['obj_type'], tags=md['tags'])
        return html_str

    def _repr_html_(self):
        md = self.read_metadata()[-1]
        html_str = """<h3> {name} </h3>""".format(name=self.name)
        html_str += StorageInterface._build_html_body_(md)
        return html_str

class HDFStorageInterface(StorageInterface):
    """
    Pandas storage interface backed by HDF5 (PyTables) for efficient
    storage of tabular data.
    """
    extension = 'hdf'
    hdf_data_level = '/data'
    hdf_format = 'fixed'

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

        if obj is not None:
            md_kwargs['index_head'] = list(str(i) for i in obj.index[:5])
            md_kwargs['length'] = len(obj)

        super(HDFStorageInterface, self).write_metadata(obj=obj, **md_kwargs)

    def _repr_html_(self):
        md = self.read_metadata()[-1]

        basic_descrip = StorageInterface._build_html_body_(md)
        extra_descrip = """
        <b>Num Entries </b> : {num_entries} <br>
        <b>Columns</b> ({n_cols}) : {col_sample} <br>
        <b>Index Head</b> : {ix_head} <br>
        """.format(num_entries=md['length'],
                   n_cols=len(md['columns']),
                   col_sample=", ".join(md['columns'][:10]),
                   ix_head=", ".join(md['index_head']))


        div_template = """
        <div style="width: 40%;">
            <div style="float:left; width: 50%">
            {basic_description}
            </div>
            <div style="float:right;">
            {extra_description}
            </div>
        </div>
        """.format(basic_description=basic_descrip,
                   extra_description=extra_descrip)

        html_str = """<h3> {name} </h3>""".format(name=self.name)
        html_str += div_template
        return html_str


#######
# Data structures to hold and map interfaces with names/extensions
storage_interfaces = dict(
    pickle=StorageInterface,
    hdf=HDFStorageInterface,
)
extension_to_interface_name_map = {v.extension: k
                                   for k, v in storage_interfaces.items()}

type_storage_lookup = {pd.DataFrame: 'hdf',
                       pd.Series: 'hdf'}

storage_type_priority_order = ['hdf', 'pickle']

########
# Hierarchical structure
# Tree -> Leaf -> Storage Types
# All one to many
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

    def __update_doc_str(self):
        docs = self.name + "\n\n"
        for st in storage_type_priority_order:
            if st not in self.type_to_storage_interface_map:
                continue

            md = self.read_metadata(storage_type=st)

            if len(md) == 0:
                docs = "No metadata!"
                break
            else:
                md = md[-1]

            auth = md.get("author")
            comm = md.get("comments")
            ts = md.get("write_time")
            ty = md.get('obj_type')
            tags = md.get('tags')

            docs += st + "\n" + "-"*len(st) + "\n"
            docs += "  Author: " + (auth if auth is not None else "No author") + "\n"
            docs += "  Timestamp: " + (ts if ts is not None else "No timestamp") + "\n"
            docs += "  Comments: " + (comm if comm is not None else "No comments") + "\n"
            docs += "  Type: " + (ty if ty is not None else "No type") + "\n"
            docs += "  Tags: " + (tags if tags is not None else "No tags") + "\n"
            docs += "\n\n"
        self.__doc__ = docs

    def __update_typed_paths(self):
        mde = idr_config['metadata_extension']
        repe = idr_config['repo_extension']

        cur_si_map = dict(self.type_to_storage_interface_map)
        cur_types = set(cur_si_map.keys())

        self.tmp = {os.path.split(p)[-1].split('.')[-1]: p
                                              for p in glob(self.save_path + '*')
                                              if p[-len(mde):] != mde
                                              and p[-len(repe):] != repe
                                              }
        self.tmp = {extension_to_interface_name_map[k]: v
                    for k, v in self.tmp.items()}

        self.type_to_storage_interface_map = {k: storage_interfaces[k](path=v, name=self.name)
                                              for k, v in self.tmp.items()}
        # Delete types that are no longer present on the FS
        next_types = set(self.type_to_storage_interface_map.keys())
        for t in (cur_types - next_types):
            si = cur_si_map[t]
            delattr(self, t)

            if hasattr(si, 'sample') and callable(getattr(si, 'sample')):
                if hasattr(self, 'sample') and self.sample == si.sample:
                   delattr(self, 'sample')

        for t in next_types:
            si = self.type_to_storage_interface_map[t]
            setattr(self, t, si)

            if hasattr(si, 'sample') and callable(getattr(si, 'sample')):
                setattr(self, 'sample', si.sample)

        self.__update_doc_str()

    def _repr_html_(self):

        for st in storage_type_priority_order:
            if st in self.type_to_storage_interface_map:
                return self.type_to_storage_interface_map[st]._repr_html_()

    def get_vector_representation_map(self):
        return {ty:si.get_vector_representation()
                for ty, si in self.type_to_storage_interface_map.items()}

    def refresh(self):
        self.type_to_storage_interface_map = dict()
        self.__update_typed_paths()

    def read_metadata(self, storage_type=None):
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

            md = self.type_to_storage_interface_map[storage_type].read_metadata()
        else:
            md = None
            for po in storage_type_priority_order:
                if po in self.type_to_storage_interface_map:
                    md = self.type_to_storage_interface_map[po].read_metadata()
                    break
        return md

    def save(self, obj, storage_type=None, auto_overwrite=False, **md_props):
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
        md_kwargs : Key-value pairs to include in the metadata entry
            Some storage interfaces may automatically extract metadata

        Returns
        -------
        None
        """
        # Need target save type for conflict detection and the eventual save
        if storage_type is None:
            storage_type = type_storage_lookup.get(type(obj), 'pickle')

        # Construct the filesystem path using a typed extension
        store_int = storage_interfaces[storage_type](self.save_path, name=self.name)

        # Double Check - file exists there or file is registered in memory
        if store_int.exists() or storage_type in self.type_to_storage_interface_map:
            if auto_overwrite:
                print("Auto overwriting '%s' (%s) in %s " % (self.name,
                                                             storage_type,
                                                             self.parent_repo.name))
            else:
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

        print("Saving to: %s.%s (%s)" % (self.parent_repo.name, self.name, storage_type))
        store_int.save(obj, **md_props)
        print("Save Complete")


        self.parent_repo._append_to_master_log(operation='save', leaf=self,
                                               author=md_props.get('author', None),
                                               storage_type=storage_type)

        self.__update_typed_paths()

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
        if storage_type is None:
            filenames = glob(self.save_path + '.*')
            print("Deleting: %s" % ",".join(filenames))
            [os.remove(fn) for fn in filenames]
        else:
            p = self.type_to_storage_interface_map[storage_type].path
            md_p = self.type_to_storage_interface_map[storage_type].md_path
            os.remove(p)
            os.remove(md_p)
        self.__update_typed_paths()
        #self.parent_repo.refresh()

        self.parent_repo._append_to_master_log(operation='delete', leaf=self,
                                               author=author,
                                               storage_type=storage_type)
        self.parent_repo.refresh()

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
        # Hack?
        self.__in_refresh = False

        #self.__build_property_tree_from_file_system()
        #self.refresh()

    def __getitem__(self, item):
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

    def __clear_property_tree(self, clear_internal_tables=False):
        for base_name, rl in self.__repo_object_table.items():
            if hasattr(self, base_name):
                delattr(self, base_name)

        for repo_name, rt in self.__sub_repo_table.items():
            if hasattr(self, repo_name):
                delattr(self, repo_name)

        if clear_internal_tables:
            self.__repo_object_table = dict()
            self.__sub_repo_table = dict()

    def __build_property_tree_from_file_system(self):
        all_dir_items = os.listdir(self.idr_prop['repo_root'])

        ### Separate out objects stored in this repo from sub-repos
        for f in all_dir_items:
            # Build listing of all base file names (no extension)
            dot_split = f.split('.')
            if not isidentifier(dot_split[0]):
                raise ValueError("File/dir name '%s' is not a valid identifier" % dot_split[0])

            base_name = dot_split[0]
            is_file = os.path.isfile(os.path.join(self.idr_prop['repo_root'], f))

            # Primary distinction - Repo (dir) vs. Obj (file)
            if is_file:
                # Objects leaf is created from base name - Leaf object will map to types
                self.add_obj_leaf(RepoLeaf(parent_repo=self, name=base_name))
            else:
                sub_repo_name = f.replace('.' + idr_config['repo_extension'], '')
                p = os.path.join(self.idr_prop['repo_root'], f)
                self.add_repo_leaf(RepoTree(repo_root=p, parent_repo=self))

        return self

    def _load_master_log(self):
        root_repo = self.get_root()
        log_name = idr_config['master_log']
        log_exists = log_name in root_repo.list(list_repos=False)
        if not log_exists:
            print("Log doesn't exist yet, creating it at %s.%s" % (self.name, log_name) )
            root_repo.save([], name=log_name, author='system',
                           comments='log of events across entire tree',
                           tags='idt_log')

        return root_repo.load(name=log_name)

    def _write_master_log(self, log_data):
        root_repo = self.get_root()
        log_name = idr_config['master_log']

        root_repo.save(log_data, name=log_name, author='system',
                       comments='log of events across entire tree',
                       tags='log', auto_overwrite=True)

    def _append_to_master_log(self, operation,
                              leaf=None, storage_type=None,
                              author=None):
        if leaf is not None:
            if leaf.name == idr_config['master_log']:
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
        """ % "\n".join("<li>%s</li>" % rt for rt in self.__sub_repo_table.keys())
        objects_html = """
        <h4>Objects</h4>
        %s
        """ % "\n".join("<li>%s</li>" % rt for rt in self.__repo_object_table.keys())

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

    def build_query_indices(self):
        parent_str = ".".join(self.get_parent_repo_names()) + '.' + self.name
        index_dict = dict()
        for name, leaf in self.__repo_object_table.items():
            k = parent_str + '.' + leaf.name
            index_dict[k] = leaf.get_vector_representation_map()

        for name, tree in self.__sub_repo_table.items():
            index_dict.update(tree.build_query_indices())

        return index_dict

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
                    self.add_obj_leaf(obj_curr[base_name])
                else:
                    self.add_obj_leaf(RepoLeaf(parent_repo=self, name=base_name))
            else:
                if base_name in repo_curr:
                    self.add_repo_tree(repo_curr[base_name])
                else:
                    p = os.path.join(self.idr_prop['repo_root'], f)
                    self.add_repo_tree(RepoTree(repo_root=p, parent_repo=self))


        curr_names = set(list(obj_curr.keys()) + list(repo_curr.keys()))
        new_names = set(list(self.__repo_object_table.keys()) + list(self.__repo_object_table.keys()))
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

    def add_obj_leaf(self, leaf):
        self.__repo_object_table[leaf.name] = leaf

        #repo_coll = hasattr(self, leaf.name) and isinstance(getattr(self, leaf.name), RepoTree)
        repo_coll = leaf.name in self.__sub_repo_table

        if not repo_coll:
            setattr(self, leaf.name, leaf)

    def add_repo_tree(self, tree):
        self.__sub_repo_table[tree.name] = tree
        setattr(self, tree.name, tree)

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
        tags : str (default=Non)
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

        leaf = self.__repo_object_table.get(name, RepoLeaf(parent_repo=self, name=name))
        leaf.save(obj, storage_type=storage_type, author=author,
                  auto_overwrite=auto_overwrite, comments=comments, tags=tags,
                  **extra_kwargs)

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

        return self.__repo_object_table[name].load()

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
            #self.__clear_property_tree()
            #self.__sub_repo_table[name] = RepoTree(repo_root=repo_root,
            #                                       parent_repo=self)
            #self.__assign_property_tree()

        self._append_to_master_log(operation='mkrepo')

        return self.__sub_repo_table[name]

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

    def summary(self):
        """
        Print basic plaintext summary of the repository
        """
        repos, objs = self.list()
        print("---Repos---")
        print("\n".join(repos))

        print("---Objects---")
        print("\n".join(objs))
