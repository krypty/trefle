# PyFUGE

TODO: explain what is PyFUGE

Then you can try to run an evolution with:

```python
from pyfuge.evo.examples import evo_cancer
evo_cancer.run()
```

## Build from sources

### Requirements

* CMake version >= 3.1
* Git
* Python >= 3.4 (tested with 3.6.5)
* To build sources on Windows you will need
  * VS Community with (you can choose which components to install using Visual Studio Installer)
    * CMake support
    * VC++ compiler
    * Windows 8.1 SDK

Note: the following instructions assume a 64bits GNU/Linux machine or a 64bits Windows machine

On Windows make sure to open an __admin cmd__ console.

1. `git clone --recursive <this_repo>`
2. `cd <this_repo>`
3. Create a virtualenv and activate it
  * On Windows
```
python -m venv myvenv
.\myenv\Scripts\activate
```
  * On Linux: __TODO__

4. If cmake is not installed you can install it with: `pip install cmake`
5. Build the sources
  * On Windows or Linux
```
cd PyFUGE
python setup.py build
```

_Note On Mac OS X, adapt the Linux instructions and don't forget to:_

```bash
brew install python
brew install gcc --without-multilib
export CC=/usr/local/bin/gcc-8
export CXX=/usr/local/bin/g++-8
pip install cmake
python setup.py build
python setup.py install
pip install wheel twine
# ...
```


## Run tests

* activate the virtualenv

```
cd PyFUGE/pyfuge/py_fiseval/FISEval
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
./run_tests --success
```

_Note: at the moment tests are only compiled on debug mode._

# Deploy to test.pypi.org

_Note you must increment the version every time you upload to (test)PyPI_

_Note you must repeat these instructions for every Python version you want to support. At the moment, PyFUGE should compile on Python >= 3.4_

## Create a binary wheel for Windows x64

Start by getting the sources as described above.

Then in the project root folder, create the binary wheel:

```bash
python setup.py build
python setup.py sdist bdist_wheel
```

Upload the binary to TestPyPI:

```bash
pip install twine
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Done ! Building a binary wheel on Windows is pretty straightforward you only need to do this once to target all Windows for a given platform (e.g. 64 bits)

## Create a binary wheel for Linux x64

As the linux ecosystem is vary diverse, an initiative called `manylinux` aims to target almost every linux systems when building Python package using C extensions.

_Note you still can run but you will obtain a wheel that might not work on other Linux systems and that will be refused by PyPI when trying to upload it. (for the latter you still can rename it to `manylinux` but this does not solve the problem...)_

```bash
python setup.py build
python setup.py sdist bdist_wheel
```

Steps to build a manylinux wheel:

1. Get the project sources and follow the requirements above
1. Install Docker and run `docker run -it --rm -v /path/to/PyFUGE:/PyFUGE quay.io/pypa/manylinux1_x86_64 /bin/bash`. _Note for an unknown reason it seems to fail on Arch Linux, had to use a Ubuntu 16.04 VM._

You will enter into a shell inside the container. The following commands are meant to be executed inside this container.
_
```bash
export PATH=/opt/python/cp36-cp36m/bin:$PATH
cd /PyFUGE
python install cmake twine
python setup.py build
python setup.py sdist bdist_wheel

# then we must fix the wheel using auditwheel
auditwheel repair dist/xxx-linux-x86_64.whl

twine upload --repository-url https://test.pypi.org/legacy/ dist/wheelhouse/*
```

## Test the uploaded binary wheel from `test.pypi.org`

```
# create an new virtualenv with a Python version that matches one of the wheel you uploaded
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyfuge==XXX_VERSION
```
