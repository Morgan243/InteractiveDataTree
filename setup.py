from setuptools import setup
setup(name='InteractiveDataTree',
      version='.1',
      #py_modules=['interactive_data_repo'],
      packages=['interactive_data_tree'],
      install_requires=['pandas', 'tables', 'mock', 'pygments']
      )

