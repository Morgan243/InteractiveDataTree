
[![Build Status](https://travis-ci.org/Morgan243/InteractiveDataTree.svg?branch=master)](https://travis-ci.org/Morgan243/InteractiveDataTree)

### What is it
DataTree implements simple file-system-based storage of Python objects
in a way that facilitates quick and simple programmatic access (i.e. interactive).
Though easily extended, the package comes with built-in support for general
objects (pickle) and Pandas objects stored in HDF. The library is Jupyter notebook
-aware and aims to be portable with limited number of dependencies.

##### Features
- Repositories, sub-repositories, and objects represented as properties in a hierarchy
    - Enables tab completion and monkey patched doc-strings with metadata
- Metadata history stored with every object
- Search metadata for particular data objects
- Portable - currently implemented as a single module with limited dependencies
- Maintainable - no special file types or database systems, no lock-in
- Flexible - Primary data storage logic is generalized in the StorageType object. Extend to add features and new types


### What's it solving
Iterative tasks (e.g. data analysis) often require more than one dataset. Furthermore, the
data may come from a variety of locations - the web, a database, unstructured data, etc - and may not be well represented
by a traditional table. Thus, practitioners are left trying to manage the source data (and their source systems) as well as
any intermediate and/or output datasets. It's a difficult and time consuming task.

DataTree aims to be the tool that bridges the gap between more advanced data storage (SQL, Hadoop, etc.)
and file system storage. The ultimate focus of this project being to make the management of many varied
datasets simple, maintainable, and portable.

### How does it work
DataTree manages the creation of a directory structure and subsequent object files. Each directory
is referred to as a 'repo', and the generic object files in these repos are mapped to DataTree Storage types.
In the common case, objects are pickle-serialized Python objects/structures, but storage types are easily
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
lvl1 = tr.mkrepo('lvl1')

# Save out a string - DataTree will default to pickle if doesn't have a better type
lvl1.save('foo bar str object', name='test_foobar')

# Flexible ways to access
print(lvl1.test_foobar.load())
print(tr['lvl1'].test_foobar.load())
```


