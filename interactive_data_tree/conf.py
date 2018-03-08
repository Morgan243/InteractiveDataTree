import os

URI_SPEC = 'datatree://'
idr_config = dict(storage_root_dir=os.path.join(os.path.expanduser("~"), '.idt_root'),
                  master_log='LOG', master_index='INDEX',
                  repo_extension='repo', metadata_extension='mdjson',
                  lock_extension='lock',
                  date_format='%h %d %Y (%I:%M:%S %p)',
                  md_vers='0.1',
                  enable_queries=True)

# Set values in this dict to be used as defaults in metadata
shared_metadata = dict(notebook_name=None, author=None, project=None)
md_fields = ['user_md', 'tree_md', 'si_md']

base_master_log_entry = dict(repo_tree=None,
                             repo_leaf=None,
                             storage_type=None,
                             repo_operation=None,
                             timestamp=None,
                             notebook_path=None, author=None)
