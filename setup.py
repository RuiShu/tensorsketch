"""Setup file for tensorbayes

For easy installation and uninstallation, do the following.

MANUAL INSTALL:
pip install .
python setup.py develop

UNINSTALL:
pip uninstall tensorsketch
"""

from setuptools import setup, find_packages
import os

setup(
    name="tensorsketch",
    version="0.0.0",
    author="Rui Shu",
    author_email="ruishu@stanford.edu",
    url="http://www.github.com/RuiShu/tensorsketch",
    # TODO(ruishu): add download_url
    # download_url="https://github.com/RuiShu/tensorsketch/archive/0.0.0.tar.gz",
    license="MIT",
    description="Lightweight deep learing library for TensorFlow 2.0",
    install_requires = [
        "numpy",
        "tensorflow"
    ],
    packages=find_packages()
)
