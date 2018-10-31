cmake_minimum_required(VERSION 3.5)
project(pyFUGE)
set(SOURCE_DIR pyfuge/py_fiseval)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable C+14 features
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCES
    ${SOURCE_DIR}/fiseval_bindings.cpp
    ${SOURCE_DIR}/trefle_fis.cpp
    )

include_directories(
    # ${SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/pyfuge/py_fiseval/include
    ${CMAKE_SOURCE_DIR}/pyfuge/py_fiseval/FISEval/include
    ${CMAKE_SOURCE_DIR}/pyfuge/py_fiseval/FISEval/vendor/eigen
    ${CMAKE_SOURCE_DIR}/pyfuge/py_fiseval/FISEval/vendor/catch2/single_include
    )

add_subdirectory(${SOURCE_DIR}/FISEval)
add_subdirectory(${SOURCE_DIR}/pybind11)
pybind11_add_module(pyfuge_c SHARED ${SOURCES} ${SOURCE_DIR}/FISEval)
target_link_libraries(pyfuge_c PRIVATE pyfuge)

message(STATUS "${SOURCES}")
