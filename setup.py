try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import data_science_toolkit


def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='data_science_toolkit',
    version="0.1.0",
    description='homework4',
    author='agbenn',
    packages=find_packages(where='', exclude=['tests']),
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/agbenn/data_science_toolkit.git',
    classifiers=[
        'Programming Language :: Python :: 3.11.4'
    ]
)
