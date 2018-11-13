cmake_minimum_required(VERSION 3.5)
project(trefle)
set(SOURCE_DIR trefle/fuzzy_engine)

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

# Compiler specific flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize -mtune=native -march=native -mavx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -static-libstdc++")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif()


set(SOURCES
    ${SOURCE_DIR}/fis_coco_eval_wrapper.cpp
    ${SOURCE_DIR}/trefle_fis.cpp
    ${SOURCE_DIR}/trefle_py_module.cpp
    )

include_directories(
    # ${SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/trefle/fuzzy_engine/include
    ${CMAKE_SOURCE_DIR}/trefle/fuzzy_engine/trefle_engine/include
    ${CMAKE_SOURCE_DIR}/trefle/fuzzy_engine/trefle_engine/vendor/catch2/single_include
    ${CMAKE_SOURCE_DIR}/trefle/fuzzy_engine/trefle_engine/vendor/json/single_include
    )

add_subdirectory(${SOURCE_DIR}/trefle_engine)
add_subdirectory(${SOURCE_DIR}/pybind11)
pybind11_add_module(trefle_engine SHARED ${SOURCES} ${SOURCE_DIR}/trefle_engine)
target_link_libraries(trefle_engine PRIVATE trefle_engine_c)

message(STATUS "${SOURCES}")
