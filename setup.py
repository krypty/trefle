from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "evo/helpers/ifs_utils.pyx",
        ["evo/helpers/ifs_utils.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
