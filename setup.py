from codecs import open
from os.path import join
import re
from setuptools import setup, find_packages

def get_version():
    versionfile = join('brancher', '__init__.py')
    lines = open(versionfile, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (versionfile,))


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='brancher',
      version=get_version(),
      description='A user-centered Python package for differentiable probabilistic inference',
      author='Brancher development team',
      author_email='info@brancher.org',
      license='MIT',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://brancher.org/",
      packages=find_packages(),
      install_requires=['python_version>="3.7"',
                        'numpy>=1.15.4',
                        'pandas>=0.23.4',
                        'matplotlib>=3.0.2',
                        'seaborn>=0.9.0',
                        'scipy>=1.1.0',
                        'tqdm>=4.28.1',
                        ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      )

__author__ = 'MindCodec'
