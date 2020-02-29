cmake_minimum_required(VERSION 3.0)
project(
matrix_solver
CXX
)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(test_mat_src "test/test_mat.cpp" "src/init.cpp" "src/matrix_op.cpp" "src/solver.cpp")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -pedantic -Wall -Wcast-qual")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")

OPTION(AVX "Use avx instructions" OFF)
OPTION(ASAN "Build with asan" OFF)
IF(AVX)
    MESSAGE("USING AVX (m % 12 == 0 must hold)")
    ADD_DEFINITIONS(-DAVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
ENDIF(AVX)
IF(ASAN)
    MESSAGE("asan")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
ENDIF(ASAN)

MESSAGE("${CMAKE_BUILD_TYPE}")

add_executable(test_mat ${test_mat_src})
target_include_directories(test_mat PRIVATE src)
target_link_libraries(test_mat m)
# cmake .. -DAVX=ON -DCMAKE_BUILD_TYPE=DEBUG
