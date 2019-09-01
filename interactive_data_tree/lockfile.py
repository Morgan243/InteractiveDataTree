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
import sys
from .conf import *


IS_PYTHON3 = sys.version_info > (3, 0)
if IS_PYTHON3:
    fs_except = FileExistsError
    prompt_input = input
else:
    prompt_input = raw_input
    fs_except = OSError


#####
class LockFile(object):
    """
    A context object (use in 'with' statement) that attempts to create
    a lock file on entry, with blocking/retrying until successful
    """
    #lock_types = ('rlock', 'wlock')
    def __init__(self, path, #lock_type='lock',
                 poll_interval=1,
                 wait_msg=True):
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

        #if lock_type not in LockFile.lock_types:
        #    raise ValueError("Unknown lock type: %s\nExpected one of %s"
        #                     % (str(lock_type), ", ".join(LockFile.lock_types)))

        #self.lock_type = lock_type

        # Make sure the directory exists
        dirs = os.path.split(path)[:-1]
        p = os.path.join(*dirs)
        valid_path = os.path.isdir(p)
        if not valid_path:
            msg = "The lock path '%s' is not valid - '%s' is not a directory"
            raise ValueError(msg % (path, p))

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()

    def unlock(self):
        os.close(self.fs_lock)
        os.remove(self.path)
        return self

    def lock(self, timeout=None):
        block_count = 0
        while True:
            try:
                self.fs_lock = os.open(self.path,
                                       os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except fs_except as e:
                block_count += 1
                if self.wait_msg:
                    print("[%d] Blocking on %s" % (block_count, self.path))
                time.sleep(self.poll_interval)
                if timeout is not None:
                    raise TimeoutError("Lock on %s timed out (%d tries)"
                                       % (self.path, timeout))

        self.locked = True
        return self

