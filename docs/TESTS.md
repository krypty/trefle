# Tests

Activate the virtualenv and:

```
cd PyFUGE/pyfuge/py_fiseval/FISEval
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
./run_tests --success
```

_Note: at the moment tests are only compiled on debug mode._
