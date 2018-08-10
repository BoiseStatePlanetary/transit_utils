from setuptools import setup

import transit_utils

version = transit_utils.__version__

setup(
    name='transit_utils',
    version=version,
    description='A small collection of methods for analyzing transit data',
    url='https://github.com/BoiseStatePlanetary/transit_utils',
    download_url='https://github.com/BoiseStatePlanetary/transit_utils/archive/'+version+'.tar.gz',
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    packages=['transit_utils'],
    keywords='',
    author='Brian Jackson',
    install_requires=['statsmodels', 'scipy', 'numpy', 'PyAstronomy'],
    author_email='bjackson@boisestate.edu'
)
