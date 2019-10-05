"""Setup file for TensorSketch.

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
    license="MIT",
    description="Lightweight deep learing library for TensorFlow 2.0",
    url="http://www.github.com/RuiShu/tensorsketch",
    packages=find_packages(),
    install_requires = ["numpy", "tensorflow"],
    author="Rui Shu",
    author_email="ruishu@stanford.edu",
)
