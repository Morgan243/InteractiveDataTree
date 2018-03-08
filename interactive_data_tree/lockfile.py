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
from interactive_data_tree.conf import *


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
        self.lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()

    def unlock(self):
        os.close(self.fs_lock)
        os.remove(self.path)
        return self

    def lock(self):
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
        return self

