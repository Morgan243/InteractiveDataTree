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
    def __init__(self, path, poll_interval=1):
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

    def is_file(self):
        return os.path.isfile(self.path)

    def load(self):
        with LockFile(self.lock_file):
            with open(self.path, mode='rb') as f:
                obj = pickle.load(f)
            return obj

    def save(self, obj, **md_kwargs):
        with LockFile(self.lock_file):
            with open(self.path, mode='wb') as f:
                pickle.dump(obj, f)

            self.write_metadata(obj=obj, **md_kwargs)

    def write_metadata(self, obj=None, **md_kwargs):
        if obj is not None:
            #md_kwargs['obj_type'] = str(type(obj))
            md_kwargs['obj_type'] = type(obj).__name__

        md_kwargs['write_time'] = datetime.now().strftime(idr_config['date_format'])

        with LockFile(self.lock_md_file):
            md = self.read_metadata(lock=False)
            md.append(md_kwargs)
            with open(self.md_path, 'w') as f:
                json.dump(md, f)

    def read_metadata(self, lock=True):
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
        raise NotImplementedError()

    def get_terms(self):
        md = self.read_metadata()
        str_md_terms = [v for k, v in md.items() if isinstance(v, str)]
        return str_md_terms

    @staticmethod
    def _build_html_body_(md):
        #md = self.read_metadata()[-1]

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
    extension = 'hdf'
    hdf_data_level = '/data'
    hdf_format = 'fixed'

    def load(self):
        with LockFile(self.lock_file):
            #hdf_store = pd.HDFStore(self.path, mode='r')
            #obj = hdf_store[HDFStorageInterface.hdf_data_level]
            obj = pd.read_hdf(self.path, mode='r')
            #hdf_store.close()
        return obj

    def save(self, obj, **md_kwargs):
        with LockFile(self.lock_file):
            hdf_store = pd.HDFStore(self.path, mode='w')
            hdf_store.put(HDFStorageInterface.hdf_data_level,
                          obj, format=HDFStorageInterface.hdf_format)
            hdf_store.close()

        self.write_metadata(obj=obj, **md_kwargs)

    def sample(self, n=5):
        with LockFile(self.lock_file):
            #hdf_store = pd.HDFStore(self.path, mode='r')
            #obj = hdf_store[HDFStorageInterface.hdf_data_level]
            obj = pd.read_hdf(self.path, mode='r', stop=n)
            #hdf_store.close()
        return obj


    def write_metadata(self, obj=None, **md_kwargs):
        if isinstance(obj, pd.DataFrame):
            md_kwargs['columns'] = list(str(c) for c in obj.columns)

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
    __metaclass__ = abc.ABCMeta
    def __init__(self, parent_repo, name):
        self.parent_repo = parent_repo
        self.name = name

        self.save_path = os.path.join(self.parent_repo.idr_prop['repo_root'], self.name)
        self.type_to_storage_interface_map = dict()
        self.__update_typed_paths()

    def __call__(self, *args, **kwargs):
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
        cur_types = set(self.type_to_storage_interface_map.keys())

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
            delattr(self, t)

        for t in next_types:
            setattr(self, t, self.type_to_storage_interface_map[t])

        self.__update_doc_str()

    def _repr_html_(self):

        for st in storage_type_priority_order:
            if st in self.type_to_storage_interface_map:
                #md = self.read_metadata(storage_type=st)[-1]
                return self.type_to_storage_interface_map[st]._repr_html_()

        #html_str = """
        #<h3> {name} </h3>
        #<b>Author</b>: {author} <br>
        #<b>Last Write</b>: {ts} <br>
        #<b>Comments</b>: {comments} <br>
        #<b>Type</b>: {ty} <br>
        #<b>Tags</b>: {tags} <br>
        #""".format(name=self.name, author=md['author'],
        #           comments=md['comments'], ts=md['write_time'],
        #           ty=md['obj_type'], tags=md['tags'])
        #return html_str

    def read_metadata(self, storage_type=None):
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
        Save Python object at this location
        """
        # Need target save type for conflict detection and the eventual save
        if storage_type is None:
            storage_type = type_storage_lookup.get(type(obj), 'pickle')

        # Construct the filesystem path using a typed extension
        store_int = storage_interfaces[storage_type](self.save_path, name=self.name)

        # Double Check - file exists there or file is registered in memory
        if store_int.is_file() or storage_type in self.type_to_storage_interface_map:
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

        self.__update_typed_paths()

    # TODO: Move delete to storage interface, perform with lock?
    def delete(self, storage_type=None):
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
        self.parent_repo.refresh()

    def load(self, storage_type=None):
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
    def __init__(self, repo_root=None, parent_repo=None):
        """
        :param repo_root: Directory that the repo is in (parent dir)
        :param repo_name: Name of the repo (.repo file in the parent dir)
        :param parent_repo: Repo Tree Object representing the parent repo
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

        self.__build_property_tree_from_file_system()

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
        in_repos = key in self.__sub_repo_table
        in_objs = key in self.__repo_object_table

        if in_objs and in_repos:
            raise ValueError("Somehow this item is in both obj and repos!?>")
        elif in_repos:
            if isinstance(value, RepoTree):
                self.__sub_repo_table[key] = value
            elif isinstance(value, RepoLeaf):
                pass
        elif in_objs:
            return self.__repo_object_table[key]
        else:
            raise KeyError("%s is not in the tree" % str(key))

    def __getattr__(self, item):

        # Only called if an unknown attribute is accessed
        # - if so, then check that the object wasn't created by another instance
        # i.e. rescan repo before throwing an error
        self.__build_property_tree_from_file_system()

        if item in self.__repo_object_table:
            return self.__repo_object_table[item]
        elif item in self.__sub_repo_table:
            return self.__sub_repo_table[item]
        else:
            dot_path = ".".join(self.get_parent_repo_names())
            dot_path = "root" if len(dot_path) == 0 else dot_path
            raise AttributeError("'%s' is not under repo %s" % (item, dot_path))

    def get_root(self):
        rep = self
        while rep.idr_prop['parent_repo'] is not None:
            rep = rep.idr_prop['parent_repo']
        return rep

    def get_parent_repo_names(self):
        rep = self#.idr_prop['parent_repo']
        repos = list()
        while rep.idr_prop['parent_repo'] is not None:
            rep = rep.idr_prop['parent_repo']
            repos.append(rep.name)

        return list(reversed(repos))

    def _load_master_log(self):
        root_repo = self.get_root()
        log_name = idr_config['master_log']
        log_exists = log_name in root_repo.list(list_repos=False)
        if not log_exists:
            print("Log doesn't exist yet, creating it at %s.%s" % (self.name, log_name) )
            root_repo.save([], name=log_name, author='system',
                           comments='log of events across entire tree',
                           tags='log')

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
        log_data = self._load_master_log()

        entry = dict(base_master_log_entry)
        entry['repo_tree'] = self.get_parent_repo_names() + [self.name]
        entry['repo_leaf'] = None if leaf is None else leaf.name
        entry['storage_type'] = None if storage_type is None else storage_type.extension
        entry['repo_operation'] = operation
        entry['timestamp'] = datetime.now()
        nbp = os.path.join(os.getcwd(), shared_metadata['notebook_name'])
        entry['notebook_path'] = nbp
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

    def __assign_property_tree(self):
        for base_name, rl in self.__repo_object_table.items():
            setattr(self, base_name, rl)

        for repo_name, rt in self.__sub_repo_table.items():
            setattr(self, repo_name, rt)

        self.__update_doc_str()

    def __build_property_tree_from_file_system(self):
        self.__clear_property_tree(clear_internal_tables=True)

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
                self.__repo_object_table[base_name] = RepoLeaf(parent_repo=self, name=base_name)
                # Repos take precedent for base name matches
                #if base_name not in self.__sub_repo_table:
                #    setattr(self, base_name, self.__repo_object_table[base_name])
            else:
                sub_repo_name = f.replace('.' + idr_config['repo_extension'], '')
                self.__sub_repo_table[sub_repo_name] = RepoTree(repo_root=os.path.join(self.idr_prop['repo_root'], f),
                                                                parent_repo=self)
                #setattr(self, sub_repo_name, self.__sub_repo_table[sub_repo_name])

        self.__assign_property_tree()
        return self

    def refresh(self):
        self.__build_property_tree_from_file_system()

    def delete(self, name, storage_type=None):
        self.__repo_object_table[name].delete(storage_type=storage_type)

    def save(self, obj, name, author=None, comments=None, tags=None,
             storage_type=None, **extra_kwargs):
        if not isidentifier(name):
            raise ValueError("Name must be a valid python identifier, got '%s'" % name)

        self.__clear_property_tree()

        leaf = self.__repo_object_table.get(name, RepoLeaf(parent_repo=self, name=name))
        leaf.save(obj, storage_type=storage_type, author=author,
                  comments=comments, tags=tags, **extra_kwargs)

        self.__repo_object_table[name] = leaf
        self.__assign_property_tree()

    def load(self, name):
        if name not in self.__repo_object_table:
            raise ValueError("Unknown object %s in repo %s" % (name, self.name))

        return self.__repo_object_table[name].load()

    def mkrepo(self, name, err_on_exists=False):
        if not isidentifier(name):
            raise ValueError("Name must be a valid python identifier, got '%s'" % name)

        repo_root = os.path.join(self.idr_prop['repo_root'], name)

        if name in self.__sub_repo_table:
            if err_on_exists:
                raise ValueError("Repo %s already exists" % repo_root)
        else:
            self.__clear_property_tree()
            self.__sub_repo_table[name] = RepoTree(repo_root=repo_root,
                                                   parent_repo=self)
            self.__assign_property_tree()
        return self.__sub_repo_table[name]

    def list(self, list_repos=True, list_objs=True, verbose=False):

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
        repos, objs = self.list()
        print("---Repos---")
        print("\n".join(repos))

        print("---Objects---")
        print("\n".join(objs))
