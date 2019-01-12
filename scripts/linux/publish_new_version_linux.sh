#!/usr/bin/env bash

# stop on first error
set -e

PYPI_USER=$1
PYPI_PASS=$2
PYVERSION=$3

# toggle comment to deploy to test.pypi.org
PYPI_REPO_URL=""
#PYPI_REPO_URL="--repository-url=https://test.pypi.org/legacy/"

PATH=/opt/python/cp${PYVERSION}-cp${PYVERSION}m/bin:$PATH

cd /PyFUGE
mkdir -p dist
# use 0.31.1 because of this issue: https://github.com/pypa/auditwheel/issues/102 
pip install -U cmake twine wheel==0.31.1
python setup.py build
python setup.py sdist bdist_wheel
cd dist
auditwheel repair *.whl
twine upload -u $PYPI_USER -p $PYPI_PASS $PYPI_REPO_URL wheelhouse/*

rm -rf /PyFUGE/dist
