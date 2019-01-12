#!/usr/bin/env bash

# must be run at the root folder PyFUGE
cd ../..
docker run -it --rm -v `pwd`:/PyFUGE trefle /bin/bash /PyFUGE/scripts/linux/publish_all_wheels_linux.sh
