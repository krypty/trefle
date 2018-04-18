import glob
import os

from setuptools import setup, find_packages, Extension

HERE = os.path.dirname(os.path.abspath(__file__))

pyfuge_module = Extension(
    'pyfuge_c',
    include_dirs=[
        os.path.join(HERE,
                     'pyfuge/cpp/FISEval/cpp/vendor/catch'),
        os.path.join(HERE,
                     'pyfuge/cpp/FISEval/cpp/vendor/eigen'),
        os.path.join(HERE,
                     'pyfuge/cpp/FISEval/cpp/vendor/pybind11/include'),
    ],
    sources=glob.glob(
        os.path.join(HERE, 'pyfuge/cpp/FISEval/cpp/src/*.cpp')),
    depends=glob.glob(
        os.path.join(HERE, 'pyfuge/cpp/FISEval/cpp/src/*.hpp')),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],

)

setup(
    name='pyfuge',
    version='0.1',
    description='PyFUGE is an educational library to play with fuzzy systems',
    author='Gary Marigliano',
    url='http://iict-space.heig-vd.ch/cpn/',
    long_description=open(os.path.join(HERE, "README.md")).read(),
    ext_modules=[pyfuge_module],
    packages=find_packages(exclude=[
        "*playground*", "*.tests", "*.tests.*", "tests.*", "tests"
    ]),
    python_requires=">3.4",
    install_requires=[
        "deap==1.2.2",
        "matplotlib==2.1.1",
        "numpy==1.14.0",
        "pandas==0.22.0",
        "scikit-learn==0.19.1",
        "scipy==1.0.0",
    ],
    setup_requires=['pytest-runner'],
    tests_require=["pytest==3.3.2"],
)
