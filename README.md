# PyFUGE

TODO

## Get the sources

_note: currently a GNU/Linux machine is required_

```bash
git clone --recurse-submodules XXX
cd pyfuge/py_fiseval/FISEval
make prepare
make
make test
```

## Build sources on Linux

Follow Get the sources section and do:

```bash
python setup.py build
python setup.py install
python setup.py test
```

Then you can try to run an evolution with:

```python
from pyfuge.evo.examples import evo_cancer
evo_cancer.run_with_simple_evo()
```

# NEW !
```bash
export VCPKG_ROOT=/home/gary/CI4CB/PyFUGE/pyfuge/py_fiseval/FISEval/vcpkg
#activate your virtualenv
python setup.py install
```
