# Deployment on PyPI

**Note** you must increment the version every time you upload to (test)PyPI

**Note** you must repeat these instructions for every Python version you want to support. At the moment, trefle should compile on Python >= 3.5

This means that for each new release you must create, at least, a:
  * Windows 64 bits binary wheel for Python 3.X
  * Linux (manylinux) 64 bits binary wheel for Python 3.X
  * Mac OS X 64 binary wheel for Python 3.X

**Note** you must repeat the 3 steps above for each Python 3.X release. So if you want to support Python 3.5, 3.6 and 3.7 you must produce 3x3 builds.

## Create a binary wheel for Windows x64

The following commands assume that you have a virtualenv using the Python version you want to.

If you don't know how to start it is recommended to use Anaconda and create n virtualenv where
n is the number of Python versions you want to support. For example, create a virtual env
called `trefle-py-35`, `trefle-py-36` and `trefle-py-37` (e.g. `conda create -n trefle-py-35 python=3.5`)

Then in the project root folder, create the binary wheel:

```bash
git clone --recursive <this_repo>
pip install wheel twine
python setup.py build
python setup.py sdist bdist_wheel
```

Upload the binary to TestPyPI:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Done ! Building a binary wheel on Windows is pretty straightforward you only need to do this once to target all Windows for a given platform (e.g. 64 bits)

## Create a binary wheel for Linux x64

As the linux ecosystem is vary diverse, an initiative called `manylinux` aims to target almost every linux systems when building Python package using C extensions.

_Note you still can run but you will obtain a wheel that might not work on other Linux systems and that will be refused by PyPI when trying to upload it. (for the latter you still can rename it to `manylinux` but this does not solve the problem...)_

Steps to build a manylinux wheel:

0. Make sure `trefle` compiles
1. Install Docker
2. `git clone --recursive <this_repo>`
3. `cd trefle`
3. Build the docker image: `cd scripts/linux && docker build -t trefle .`
4. `cd ..` You should now be in `trefle` folder.
5. Install Docker and run `docker run -it --rm -v $PWD:/PyFUGE trefle`. _Note for an unknown reason it seems to fail on Arch Linux, had to use a Ubuntu 16.04 VM._

You will be prompted to enter your PyPI credentials to upload the wheels.

## Instructions to deploy on Mac OS X

Open a terminal and:

```bash
git clone --recursive <this_repo>
cd trefle
brew update
brew install pyenv
brew install gcc --without-multilib
pyenv install 3.5.6
pyenv install 3.6.7
pyenv install 3.7.1
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile # then restart the terminal
```
Then for each Python P_XXX version in (3.5.6, 3.6.7, 3.7.1) do:

```bash
# create a virtualenv for this version (only do it once per version)
pyenv virtualenv P_XXX trefle-P_XXX # e.g. trefle-3.5.6

# activate the virtual env (this and the following commands have to be done
# every time)
pyenv activate trefle-P_XXX

# check the version with `python -V`
export CC=/usr/local/bin/gcc-8
export CXX=/usr/local/bin/g++-8
pip install cmake wheel twine
python setup.py build
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ wheelhouse/*
rm -rf build dist
```

## Test the uploaded binary wheel from `test.pypi.org`

```
# create an new virtualenv with a Python version that matches one of the wheel you uploaded
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple trefle==XXX_VERSION
```


