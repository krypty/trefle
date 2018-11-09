#!/usr/bin/env bash

# must be run at the root folder PyFUGE
docker run -it --rm -v `pwd`:/PyFUGE quay.io/pypa/manylinux1_x86_64 /bin/bash /PyFUGE/scripts/linux/publish_all_wheels_linux.sh"
