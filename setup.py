import codecs
import os
from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='activevision',
    version=get_version("activevision/__init__.py"),
    author='Sulabh Shrestha',
    author_email='sulabh.shr@gmail.com',
    description='Utilities package for Active Vision Dataset',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/sulabh-shr/activevision.git',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas'],
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)