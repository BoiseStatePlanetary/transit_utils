from setuptools import setup

version = '0.0.16'

setup(
    name='transit_utils',
    version=version,
    description='A small collection of methods for analyzing transit data',
    long_description=long_description,
    url='https://github.com/BoiseStatePlanetary/transit_utils',
    download_url='https://github.com/BoiseStatePlanetary/transit_utils/archive/'+version+'.tar.gz',
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    include_package_data=True,
    author='Brian Jackson',
    install_requires=['statsmodels', 'scipy', 'numpy', 'PyAstronomy'],
    author_email='bjackson@boisestate.edu'
)
