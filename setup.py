# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

install_requires = [
    "decorator==4.4.2",
    "matplotlib",
    "openfoamparser",
    "plotly",
    "pyfoam",
    "pytorch_lightning",
    "torch",
    "torchvision",
    "torch_geometric",
    "torch-geometric-temporal",
    "torch_sparse",
    "torch_scatter",
]

test_requires = [
    'pytest',
    'pytest-cov',
]

setup(
    name="foamgraph",
    version="0.1.0",
    description="FoamGraph is a Python library for manipulating OpenFOAM cases as graphs.",
    author="Gabriel Goncalves",
    author_email="",
    url="https://github.com/",
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        'test': test_requires,
    },
    packages=find_packages(exclude=("tests", "docs")),
)
