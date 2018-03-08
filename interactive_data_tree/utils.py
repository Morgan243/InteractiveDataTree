import sys
import warnings
from ast import parse
from .conf import *

def is_valid_uri(obj):
    if not isinstance(obj, basestring):
        return False
    elif obj[:len(URI_SPEC)] == URI_SPEC:
        return True
    else:
        return False

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

try:
    import tables
    original_warnings = list(warnings.filters)
    warnings.simplefilter('ignore', tables.NaturalNameWarning)
except ImportError as e:
    print("Unable to import PyTables (tables) - can't use HDF dataframe storage!")

def isidentifier(name):
    try:
        parse('{} = None'.format(name))
        return True
    except (SyntaxError, ValueError, TypeError) as e:
        return False

def reference_to_leaf(tree, obj):
    if isinstance(obj, basestring) and is_valid_uri(obj):
        return tree.from_reference(obj)
    elif isinstance(obj, basestring):
        return obj
    else:
        return obj

def md_resolve(tree, t_md):
    for k in t_md.keys():
        t_md[k] = reference_to_leaf(tree, t_md[k])
    return t_md
