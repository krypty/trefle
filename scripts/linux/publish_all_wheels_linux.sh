#!/usr/bin/env bash

# stop on first error
set -e

#PYPI_URL="https://pypi.org/legacy/"
PYPI_URL="https://test.pypi.org/legacy/"

echo -n PyPI username:
read PYPI_USER

echo -n PyPI password:
read -s PYPI_PASS

for pyversion in 35 36 37
do
	echo "\n\n"
	echo "***************************************************"
	echo "*** Publish new version for Python $pyversion...***"
	echo "***************************************************"
	bash /PyFUGE/scripts/linux/publish_new_version_linux.sh $PYPI_USER $PYPI_PASS $pyversion $PYPI_URL
done
