from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='brancher',
      version='0.1.0',
      description='Write description',
      author='MindCodec',
      author_email='something@mindcodec.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/LucaAmbrogioni/Brancher",
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
