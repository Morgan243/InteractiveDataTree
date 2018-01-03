### What is it
The DataTree implements simple file-system based storage of Python objects
in a way that facilitates quick and simple programmatic access (i.e. interactive).
Built-in support for general objects (pickle) and Pandas objects stored in HDF.

```python
import interactive_data_tree as tr

# Repository Tree Root defaults to ~/.idr_root.repo
rt = tr.RepoTree()

# Make a new Sub-repo
lvl1 = tr.mkrepo('lvl1')

# Save out a string - DataTree will default to pickle if doesn't have a better type
lvl1.save('foo bar str object', name='test_foobar')

# Flexible ways to accessibility
print(lvl1.test_foobar.load())
print(tr['lvl1'].test_foobar.load())
```