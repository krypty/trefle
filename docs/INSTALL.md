# Installation

## From PyPI (recommended)

**Work In Progress please see From sources section**

```bash
pip install pyfuge
```

Verify that it works with a Python terminal:

```python
from pyfuge.evo.examples import evo_cancer
evo_cancer.run()
```

## From sources

* CMake version >= 3.1
* Git
* Python >= 3.4 (tested with 3.6.5)
* To build sources on Windows you will need
  * Visual Studio Community with (you can choose which components to install using Visual Studio Installer)
    * CMake support
    * VC++ compiler
    * Windows 8.1 SDK

Note: the following instructions assume a 64bits GNU/Linux machine or a 64bits Windows machine

### Build instructions for Windows

On Windows make sure to open an __admin cmd__ console.

1. `git clone --recursive <this_repo>`
2. `cd <this_repo>`
3. Create a virtualenv and activate it
```
python -m venv myvenv
.\myenv\Scripts\activate
```
4. If cmake is not installed you can install it with: `pip install cmake`
5. Build the sources
```
cd PyFUGE
python setup.py build
```
6. Install PyFUGE inside your virtualenv with `python setup.py install`


### Build instructions for GNU/Linux

1. `git clone --recursive <this_repo>`
2. `cd <this_repo>`
3. Create a virtualenv and activate it
```
python -m venv myvenv
./myenv/bin/activate
```
4. If cmake is not installed you can install it with: `pip install cmake`
5. Build the sources
```
cd PyFUGE
python setup.py build
```
6. Install it inside your virtualenv with `python setup.py install`


### Build instructions for Mac OS X 64 bits

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
```
