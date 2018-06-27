#!/usr/bin/env bash

# stop on first error
set -e

PYPI_USER=$1
PYPI_PASS=$2
PYVERSION=$3

PATH=/opt/python/cp${PYVERSION}-cp${PYVERSION}m/bin:$PATH

cd /PyFUGE
mkdir -p dist
pip install cmake twine wheel
python setup.py build
python setup.py sdist bdist_wheel
cd dist
auditwheel repair *.whl
twine upload -u $PYPI_USER -p $PYPI_PASS --repository-url "https://test.pypi.org/legacy/" wheelhouse/*

rm -rf /PyFUGE/dist
