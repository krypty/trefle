FROM quay.io/pypa/manylinux1_x86_64

VOLUME ["/PyFUGE"]


RUN curl -L https://github.com/Noctem/pogeo-toolchain/releases/download/v1.4/gcc-7.2-binutils-2.29-centos5-x86-64.tar.bz2 -o toolchain.tar.bz2

RUN tar -C / -xf toolchain.tar.bz2

ENV MAKEFLAGS=-j4
ENV LD_LIBRARY_PATH "/toolchain/lib64:/toolchain/lib:${LD_LIBRARY_PATH}"
ENV MFLAG "-m64"
ENV MANYLINUX 1
ENV PATH "/toolchain/bin:${PATH}"
ENV CFLAGS "-I/toolchain/include ${MFLAG}"
ENV CXXFLAGS "-I/toolchain/include ${MFLAG}"

#ENV PATH "/opt/python/cp35-cp35m/bin:${PATH}"

#RUN pip install cmake twine wheel

CMD ["/bin/bash", "/PyFUGE/scripts/linux/publish_all_wheels_linux.sh"]


