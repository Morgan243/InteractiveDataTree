
[![Build Status](https://travis-ci.org/Morgan243/InteractiveDataTree.svg?branch=master)](https://travis-ci.org/Morgan243/InteractiveDataTree)

[![Coverage Status](https://coveralls.io/repos/github/Morgan243/InteractiveDataTree/badge.svg?branch=master)](https://coveralls.io/github/Morgan243/InteractiveDataTree?branch=master)

### What is it
DataTree implements a simple file-system-based storage of Python objects
in a way that facilitates quick and simple programmatic access (i.e. interactive).
Functionally, the DataTree builds a dot representation (attribute access on objects)
based on files and directories. Directories are 'Repos' (repositories) and files
are mapped onto a DataTree StorageInterface based on the file's extension
(e.g. pickle -> '.pkl').

The package comes with built-in support for general objects (pickle) and Pandas
objects stored in HDF. The library is Jupyter notebook-aware and will use HTML
representations where sensible.

##### Features
- Repositories, sub-repositories, and objects represented as properties on a tree object
    - Enables *tab completion*, monkey patched *doc-strings with metadata*, and *rich HTML representations*
- **Maintainable** - no special internal data or database systems, no lock-in.
- **Flexible** - Primary data storage logic is generalized in the StorageType object. Extend to add features and new types
- **Metadata history** stored with every object and made *searchable*


### What's it solving
Iterative tasks in data analysis often require more than one dataset. Furthermore, the
data may come from a variety of locations - the web, a database, unstructured data, etc - and may not be well represented
by a traditional table. Thus, practitioners are left trying to manage the source data (and their source systems) as well as
any intermediate and/or output datasets. It's a difficult and time consuming task.

The ultimate focus of this project being to make the management of many varied
datasets simple, maintainable, and portable. The only expectation for use with DataTree is that the
the data can be represented as data that can be stored on the local filesystems. For standarad datasets,
this likely means storing the data itself in the DataTree. However, new interfaces can be implemented
that simply store the information required to access a remote system (e.g. store a JSON file with connection
information and SQL query - retrieve data on load).

### How does it work
DataTree manages the creation of directories and subsequent object files. Each directory
is referred to as a 'repo', and the generic object files in these repos are mapped to DataTree StorageTypes.
In the common case, objects are pickle-serialized Python objects, but storage types are easily
extended. An object name may have several different types stored, so namespaces across types won't collide.

Every object has metadata stored alongside it in a JSON file. Each storage type can
choose to use the default metadata storage, amend the default storage by overriding the write
procedure, or implement and entirely different metadata storage all together.

### Example

```python
import interactive_data_tree as idt

# Repository Tree Root defaults to ~/.idt_root.repo
tr = idt.RepoTree()

# Make a new Sub-repo
# - This creates the directory ~/.idt_root.repo/lvl1.repo
lvl1 = tr.mkrepo('lvl1')

# Save out a string - DataTree will default to pickle if doesn't have a better type
# - This writes the file ~/.idt_root.repo/lvl1.repo/test_foobar.pkl
# - Metadata stored in ~/.idt_root.repo/lvl1.repo/test_foobar.pkl.mdjson
lvl1.save('foo bar str object', name='test_foobar')

# Flexible ways to access
assert lvl1 == tr.lvl1
print(lvl1.test_foobar.load())
print(tr['lvl1'].test_foobar.load())
```


