from glob import glob
import json
import pickle
import pandas as pd
import os


import keyword
import tokenize

# From: https://stackoverflow.com/questions/12700893/how-to-check-if-a-string-is-a-valid-python-identifier-including-keyword-check
def isidentifier(ident):
    """Determines if string is valid Python identifier."""

    # Smoke test — if it's not string, then it's not identifier, but we don't
    # want to just silence exception. It's better to fail fast.
    if not isinstance(ident, str):
        raise TypeError("expected str, but got {!r}".format(type(ident)))

    # Quick test — if string is in keyword list, it's definitely not an ident.
    if keyword.iskeyword(ident):
        return False

    readline = (lambda: (yield ident.encode('utf-8-sig')))().__next__
    tokens = list(tokenize.tokenize(readline))

    # You should get exactly 3 tokens
    if len(tokens) != 3:
        return False

    # First one is ENCODING, it's always utf-8 because we explicitly passed in
    # UTF-8 BOM with ident.
    if tokens[0].type != tokenize.ENCODING:
        return False

    # Second is NAME, identifier.
    if tokens[1].type != tokenize.NAME:
        return False

    # Name should span all the string, so there would be no whitespace.
    if ident != tokens[1].string:
        return False

    # Third is ENDMARKER, ending stream
    if tokens[2].type != tokenize.ENDMARKER:
        return False

    return True


idr_config = dict(storage_root_dir='/home/morgan/.idr_root.repo',
                  repo_extension='repo')

# IDEAS
# - Most recent (list most recently accessed/written)
#

#######
hdf_file_extension = '.hdf'
hdf_data_level = '/data'
metadata_extension = 'mdjson'

#####
def read_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

type_storage_lookup = {pd.DataFrame: 'hdf',
                       pd.Series: 'hdf'}

storage_type_priority_order = ['hdf', 'pickle']

class StorageInterface(object):
    extension = 'pkl'
    def __init__(self, path):
        if path[-len(self.extension):] != self.extension:
            self.path = path + '.' + self.extension
        else:
            self.path = path

        self.md_path = self.path + '.' + metadata_extension

    def is_file(self):
        return os.path.isfile(self.path)

    def read(self):
        with open(self.path, mode='rb') as f:
            obj = pickle.load(f)
        return obj

    def write(self, obj):
        with open(self.path, mode='wb') as f:
            pickle.dump(obj, f)

    def write_metadata(self, **md_kwargs):
        md = self.read_metadata()
        md.append(md_kwargs)
        write_json(md, self.md_path)

    def read_metadata(self):
        if os.path.isfile(self.md_path):
            md = read_json(self.md_path)
        else:
            md = []
        return md

class HDFStorageInterface(StorageInterface):
    extension = 'hdf'

    def read(self):
        hdf_store = pd.HDFStore(self.path, mode='r')
        obj = hdf_store[hdf_data_level]
        hdf_store.close()
        return obj

    def write(self, obj):
        hdf_store = pd.HDFStore(self.path, mode='w')
        hdf_store.put(hdf_data_level, obj, format='fixed')
        hdf_store.close()

storage_interfaces = dict(
    pickle=StorageInterface,
    hdf=HDFStorageInterface,
)
extension_to_interface_name_map = {v.extension: k
                                   for k, v in storage_interfaces.items()}

class RepoLeaf(object):
    def __init__(self, parent_repo, name):
        self.parent_repo = parent_repo
        self.name = name

        self.save_path = os.path.join(self.parent_repo.idr_prop['repo_root'], self.name)
        self.md_path = self.save_path + metadata_extension
        self.__update_typed_paths()

    def __call__(self, *args, **kwargs):
        return self.load(storage_type=None)

    def __update_typed_paths(self):
        self.typed_path_map = {os.path.split(p)[-1].split('.')[-1]: p
                               for p in glob(self.save_path + '*')
                               if p[-len(metadata_extension):] != metadata_extension
                               and p[-len(idr_config['repo_extension']):] != idr_config['repo_extension']
                               }
        self.typed_path_map = {extension_to_interface_name_map[k]: v
                               for k, v in self.typed_path_map.items()}

        self.typed_path_map = {k: storage_interfaces[k](path=v)
                               for k, v in self.typed_path_map.items()}

    def read_metadata(self, storage_type=None):
        if storage_type is not None:
            if storage_type not in self.typed_path_map:
                raise ValueError("Type %s does not exist for %s" % (storage_type, self.name))

            md = self.typed_path_map[storage_type].read_metadata()
        else:
            for po in storage_type_priority_order:
                if po in self.typed_path_map:
                    md = self.typed_path_map[po].read_metadata()
                    break
        return md

    def save(self, obj, storage_type=None, **md_props):
        """
        Save Python object at this location
        """
        # Need target save type for conflict detection and the eventual write
        if storage_type is None:
            storage_type = type_storage_lookup.get(type(obj), 'pickle')

        # Construct the filesystem path using a typed extension
        store_int = storage_interfaces[storage_type](self.save_path)

        # Double Check - file exists there or file is registered in memory
        if store_int.is_file() or storage_type in self.typed_path_map:
            prompt = "An object named '%s' (%s) in %s already exists" % (self.name,
                                                                         storage_type,
                                                                         self.parent_repo.name)
            prompt += "Overwrite this object? (y/n)"
            y = input(prompt)
            if y == 'y' or y == 'yes':
                print("Proceeding with overwrite...")
            else:
                print("Aborting...")
                return

        print("Saving to: %s.%s (%s)" % (self.parent_repo.name, self.name, storage_type))
        store_int.write(obj)
        store_int.write_metadata(**md_props)
        print("Save Complete")

        self.__update_typed_paths()

    def delete(self):
        filenames = glob(self.save_path + '.*')
        print("Deleting: %s" % ",".join(filenames))
        [os.remove(fn) for fn in filenames]
        self.__update_typed_paths()
        self.parent_repo.refresh()

    def load(self, storage_type=None):
        store_int = None
        if storage_type is None:
            for po in storage_type_priority_order:
                if po not in self.typed_path_map:
                    continue

                store_int = self.typed_path_map[po]
        else:
            store_int = self.typed_path_map[storage_type]

        return store_int.read()

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

    def __clear_property_tree(self, clear_internal_tables=False):
        for base_name, rl in self.__repo_object_table.items():
            delattr(self, base_name)

        for repo_name, rt in self.__sub_repo_table.items():
            delattr(self, repo_name)

        if clear_internal_tables:
            self.__repo_object_table = dict()
            self.__sub_repo_table = dict()


    def __assign_property_tree(self):
        for base_name, rl in self.__repo_object_table.items():
            setattr(self, base_name, rl)

        for repo_name, rt in self.__sub_repo_table.items():
            setattr(self, repo_name, rt)


    def __build_property_tree_from_file_system(self):
        self.__clear_property_tree(clear_internal_tables=True)

        all_dir_items = os.listdir(self.idr_prop['repo_root'])

        ### Separate out objects stored in this repo from sub-repos
        for f in all_dir_items:
            # Build listing of all base file names (no extension)
            base_name = f.split('.')[0]
            is_file = os.path.isfile(os.path.join(self.idr_prop['repo_root'], f))

            # Primary distinction - Repo (dir) vs. Obj (file)
            if is_file:
                # Objects leaf is created from base name - Leaf object will map to types
                self.__repo_object_table[base_name] = RepoLeaf(parent_repo=self, name=base_name)
                # Repos take precedent for base name matches
                if base_name not in self.__sub_repo_table:
                    setattr(self, base_name, self.__repo_object_table[base_name])
            else:
                sub_repo_name = f.replace('.' + idr_config['repo_extension'], '')
                self.__sub_repo_table[sub_repo_name] = RepoTree(repo_root=os.path.join(self.idr_prop['repo_root'], f))
                setattr(self, sub_repo_name, self.__sub_repo_table[sub_repo_name])
        return self


    def refresh(self):
        self.__build_property_tree_from_file_system()

    def save(self, obj, name, author=None, comments=None, tags=None,
             storage_type=None, **extra_kwargs):
        if not isidentifier(name):
            raise ValueError("Name must be a valid python identifier, got '%s'" % name)

        self.__clear_property_tree()

        leaf = RepoLeaf(parent_repo=self, name=name)
        leaf.save(obj, storage_type=storage_type, author=author,
                  comments=comments, tags=tags, **extra_kwargs)

        self.__repo_object_table[name] = leaf
        self.__assign_property_tree()
        #self.__build_property_tree_from_file_system()

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
