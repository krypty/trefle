#!/usr/bin/env bash

# stop on first error
set -e

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
	bash /PyFUGE/scripts/linux/publish_new_version_linux.sh $PYPI_USER $PYPI_PASS $pyversion
done
