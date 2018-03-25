import math
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
from .lockfile import LockFile
from .utils import *

standard_metadata = ['author', 'comments', 'tags',
                     'write_time', 'obj_type']
############################
class Metadata(object):
    metadata_port = lambda x: x
    def __init__(self, path, lock_path=None, required_fields=None,
                 resolve_tree=None):
        self.path = path
        self.lock_path = path + '.' + idr_config['lock_extension'] \
            if lock_path is None else lock_path
        self.required_fields = list() if required_fields is None else required_fields
        self.resolve_tree = resolve_tree

    def __getitem__(self, item):
        md = self.read_metadata(most_recent=True)
        md['user_md'] = md_resolve(self.resolve_tree,
                                   md.get('user_md', dict()))
        md['si_md'] = md_resolve(self.resolve_tree,
                                 md.get('si_md', dict()))

        # Order of precedence, in reverse (always have user keys)
        ret = md.get('tree_md', dict())
        ret.update(md['si_md'])
        ret.update(md['user_md'])
        return ret[item]

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
            # Go through top level : tree_md, si_md, user_md
            for k in md_d.keys():
                # Get the current 'latest' entry
                tmp = latest_md.get(k, dict())

                if md_d[k] is not None and len(md_d[k]) > 0:
                    tmp.update(md_d[k])
                latest_md[k] = tmp
        return latest_md

    def get_missing_metadata_fields(self, md, required_metadata):
        if not isinstance(md, dict):
            raise ValueError("Expected dict, got %s" % str(type(md)))

        if required_metadata is None:
            req_md = list()
        elif not isinstance(required_metadata, list):
            req_md = [self.required_fields]
        else:
            req_md = self.required_fields

        missing = list()
        for rm in req_md:
            if rm not in md:
                missing.append(rm)
        return missing

    def touch(self):
        if not self.exists():
            with LockFile(self.lock_path):
                f = open(self.path, 'w')
                f.write('')
                f.close()
        return self

    def exists(self):
        return os.path.isfile(self.path)

    def remove(self):
        with LockFile(self.lock_path):
            os.remove(self.path)

    def write_metadata(self, **kwargs):
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

        with LockFile(self.lock_path):
            md = self.read_metadata(most_recent=False, lock=False)
            most_recent_md = Metadata.__collapse_metadata_deltas(md)

            # Remove key values that we already have stored (key AND value match)
            for f in md_fields:
                curr_field_map = most_recent_md.get(f, dict())
                for field_k, field_v in curr_field_map.items():
                    got_new_val = field_k in kwargs
                    val_is_same = field_v == kwargs.get(f, dict()).get(field_k)
                    if got_new_val and val_is_same:

                        del kwargs[f][field_k]

            md.append(kwargs)
            with open(self.path, 'w') as f:
                json.dump(md, f)

    def read_metadata(self, most_recent=True, lock=True):
        if not self.exists():
            md = [dict(tree_md=dict(md_vers=idr_config['md_vers']),
                       user_md=dict(),
                       si_md=dict())]
        else:
            if lock:
                with LockFile(self.lock_path):
                    with open(self.path, 'r') as f:
                        md = json.load(f)
            else:
                with open(self.path, 'r') as f:
                    md = json.load(f)

            #if md[0].get('tree_md', dict()).get('md_vers', dict()) != idr_config['md_vers']:
            #    md = metadata_port(md)
            md = Metadata.metadata_port(md)

        if most_recent:
            md = Metadata.__collapse_metadata_deltas(md)

        return md

    def add_reference(self, reference_uri, *reference_md_keys):
        with LockFile(self.lock_path):
            md_hist = self.read_metadata(most_recent=False, lock=False)
            last_md = self.__collapse_metadata_deltas(md_hist)
            references = last_md['tree_md'].get('references', dict())
            keys = references.get(reference_uri, list())

            references[reference_uri] = list(set(keys + list(reference_md_keys)))
            for k in references[reference_uri]:
                last_md['user_md'][k] = reference_uri

            last_md['tree_md']['references'] = references
            md_hist[-1] = last_md

            with open(self.path, 'w') as f:
                json.dump(md_hist, f)

    def remove_reference(self, reference_uri):
        with LockFile(self.lock_path):
            md_hist = self.read_metadata(most_recent=False, lock=False)
            last_md = self.__collapse_metadata_deltas(md_hist)
            references = last_md['tree_md'].get('references', dict())

            if reference_uri not in references:
                msg = "URI '%s' not in '%s' references" % (reference_uri, 'UNK')
                raise ValueError(msg)

            keys = references[reference_uri]
            for k in keys:
                last_md['user_md'][k] = '[DELETED]' + str(last_md['user_md'][k])

            del references[reference_uri]
            last_md['tree_md']['references'] = references
            md_hist[-1] = last_md

            with open(self.path, 'w')  as f:
                json.dump(md_hist, f)

    def add_referrer(self, referrer_uri, *referrer_md_keys):
        with LockFile(self.lock_path):
            md_hist = self.read_metadata(most_recent=False, lock=False)
            last_md = self.__collapse_metadata_deltas(md_hist)

            # Referrers maps a leaf URI to keys in the metadata
            referrers = last_md['tree_md'].get('referrers', dict())

            if referrer_uri not in referrers:
                referrers[referrer_uri] = list()

            # Add keys that were set to this reference, use set to remove duplicates
            referrers[referrer_uri].extend(referrer_md_keys)
            referrers[referrer_uri] = list(set(referrers[referrer_uri]))

            last_md['tree_md']['referrers'] = referrers
            md_hist[-1] = last_md

            with open(self.path, 'w')  as f:
                json.dump(md_hist, f)

    def remove_referrer(self, referrer_uri):
        with LockFile(self.lock_path):
            md_hist = self.read_metadata(most_recent=False, lock=False)
            last_md = self.__collapse_metadata_deltas(md_hist)

            if 'referrers' not in last_md['tree_md']:
                raise ValueError("Cannot remove a referrer that doesn't exist")

            referrers = last_md['tree_md']['referrers']

            if referrer_uri not in referrers:
                raise ValueError("URI '%s' is not a referrer to %s" %
                                 (referrer_uri, 'FIXME'))

            del referrers[referrer_uri]

            #for k in referrer_md_keys:
            #    referrers[referrer_uri].remove(k)

            #if len(referrers[referrer_uri]) == 0:
            #    del referrers[referrer_uri]

            last_md['tree_md']['referrers'] = referrers
            md_hist[-1] = last_md

            with open(self.path, 'w') as f:
                json.dump(md_hist, f)
