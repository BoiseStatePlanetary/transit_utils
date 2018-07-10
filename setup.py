from setuptools import setup, find_packages
from codecs import open
from os import path

import transit_utils

__version__ = transit_utils.__version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='transit_utils',
    version=__version__,
    description='A small collection of methods for analyzing transit data',
    long_description=long_description,
    url='https://github.com/BoiseStatePlanetary/transit_utils',
    download_url='https://github.com/BoiseStatePlanetary/transit_utils/archive/'+__version__+'.tar.gz',
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    include_package_data=True,
    author='Brian Jackson',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='bjackson@boisestate.edu'
)
