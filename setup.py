import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from glob import glob
from shutil import copyfile, copymode

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            # build_args += ['--', '-j2']

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        # Copy *.pyd file to build directory since VS write it somewhere else
        if platform.system() == "Windows":
            lib_file = glob(os.path.join(self.build_temp, cfg, "*.pyd"))[0]
            dest_file = os.path.join(extdir, lib_file.split(os.sep)[-1])
            copyfile(lib_file, dest_file)
        print()  # Add an empty line for cleaner output

    def copy_test_file(self, src_file):
        """
        Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.
        By default, message like `creating directory /path/to/package` and
        `copying directory /src/path/to/package -> path/to/package` are
        displayed on standard output. Adapted from scikit-build.
        """
        # Create directory if needed
        dest_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tests", "bin"
        )
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy file
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        print("copying {} -> {}".format(src_file, dest_file))
        copyfile(src_file, dest_file)
        copymode(src_file, dest_file)


trefle_module = CMakeExtension("trefle")

setup(
    name="trefle",
    version="0.2",
    description="Trefle stands for Trefle is a Revised and Evolutionary-based "
    "Fuzzy Logic Engine. It is an implementation of the FuzzyCoCo "
    "algorithm i.e. a scikit-learn compatible estimator that use "
    "a cooperative coevolution algorithm to find and build "
    "interpretable fuzzy systems. Designed for both students and "
    "researchers ",
    author="Gary Marigliano",
    url="http://iict-space.heig-vd.ch/cpn/",
    long_description=open(os.path.join(HERE, "README.md")).read(),
    long_description_content_type="text/markdown",
    ext_modules=[trefle_module],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(
        exclude=["*playground*", "*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    python_requires=">=3.5",
    install_requires=[
        "deap>=1.2.2",
        "pandas>=0.22.0",
        "scikit-learn>=0.19.1",
        "scipy>=1.0.0",
        "numpy>=1.14.0",
        "bitarray==0.8.3",
    ],
    extras_require={"evo_plot": ["matplotlib>=2.1.1"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest==3.3.2"],
    zip_safe=False,
    include_package_data=True,
)
