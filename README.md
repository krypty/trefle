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
* vcpkg
* Python >= 3.4 (tested with 3.6.5)
* To build sources on Windows you will need
  * VS Community with (you can choose which components to install using Visual Studio Installer)
    * CMake support
    * VC++ compiler
    * Windows 8.1 SDK

Note: the following instructions assume a 64bits GNU/Linux machine or a 64bits Windows machine

On Windows make sure to open an __admin cmd__ console.

1. Download [vcpkg](https://github.com/Microsoft/vcpkg):
  * git clone it somewhere. It does not matter where
  * cd vcpkg
  * execute bootstrap-vcpkg.{sh,bat}
  * set `VCPKG_ROOT` environment variable
    * Windows: `set VCPKG_ROOT=X:\path\to\vcpkg`
    * Linux: `export VCPKG_ROOT=/path/to/vcpkg`
2. Install vcpkg dependencies
  * On Windows
```
cd %VCPKG_ROOT%
.\vcpkg.exe install Eigen3:x64-windows
.\vcpkg.exe install Catch2:x64-windows
.\vcpkg.exe install pybind11:x64-windows
```
 * On Linux
```
cd $VCPKG_ROOT
./vcpkg install Eigen3
./vcpkg install Catch2
./vcpkg install pybind11
```

3. Create a virtualenv and activate it
  * On Windows
```
python -m venv myvenv
.\myenv\Scripts\activate
```
  * On Linux: __TODO__

4. Clone this repository

```
git clone --recursive $this_repo$
cd PyFUGE
```


5. Build the sources
  * On Windows or Linux
```
cd PyFUGE
python setup.py build
```


## Run tests

* activate the virtualenv
* export the `VCPKG_ROOT` environment variable

```
cd PyFUGE/pyfuge/py_fiseval/FISEval
mkdir build
cd build
cmake "-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
./run_tests --success
```

Note: at the moment tests are only compile on debug mode.
