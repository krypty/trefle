import os
import subprocess

RETURN_CODE_OK = 0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_PATH = os.path.join(HERE, os.pardir)

TREFLE_ENGINE_DIR = os.path.join(
    ROOT_PROJECT_PATH, "trefle", "fuzzy_engine", "trefle_engine"
)

BUILD_DIR = os.path.join(TREFLE_ENGINE_DIR, "build")
BUILD_DIR = os.path.normpath(BUILD_DIR)


def test_run_trefle_engine_tests():
    run_cmake_with_debug_flag()
    run_trefle_engine_tests()


def run_cmake_with_debug_flag():
    cmake_setup_args = ["-DCMAKE_BUILD_TYPE=Debug", ".."]
    cmake_build_args = ["--build", "."]

    subprocess.check_output(["cmake"] + cmake_setup_args, cwd=BUILD_DIR)
    subprocess.check_output(["cmake"] + cmake_build_args, cwd=BUILD_DIR)


def run_trefle_engine_tests():
    subprocess.check_call(["./run_tests", "--success"], cwd=BUILD_DIR)
