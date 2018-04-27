cmake_minimum_required(VERSION 3.1)
project(pyFUGE)
set(SOURCE_DIR pyfuge/py_fiseval)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 REQUIRED NO_MODULE)
find_package(OpenMP)
if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(
    # ${SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/pyfuge/py_fiseval/FISEval/include
    ${EIGEN3_INCLUDE_DIR}
    )
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
add_subdirectory(${SOURCE_DIR}/FISEval)
add_subdirectory(${SOURCE_DIR}/pybind11)
pybind11_add_module(pyfuge_c SHARED "${SOURCE_DIR}/fiseval_bindings.cpp" ${SOURCE_DIR}/FISEval)
target_link_libraries(pyfuge_c PRIVATE pyfuge)

message(STATUS "${SOURCES}")
