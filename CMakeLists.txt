cmake_minimum_required(VERSION 3.0)
find_package(Threads REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)


set(test_mpi_src "test/test_mpi.cpp" )
set(test_mat_src "test/test_mat.cpp" "src/init.cpp" "src/matrix_op.cpp" "src/solver.cpp")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -pedantic -Wall -Wcast-qual")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")
OPTION(AVX "Use avx instructions" OFF)
IF(AVX)
    MESSAGE("USING AVX")
    ADD_DEFINITIONS(-DAVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
ENDIF(AVX)
IF(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    MESSAGE("debug version with asan")
ENDIF(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")

# add_executable(test_mpi ${test_mpi_src})
add_executable(test_mat ${test_mat_src})
target_include_directories(test_mat PRIVATE src)
target_link_libraries(test_mat m)
# cmake .. -DAVX=ON -DCMAKE_BUILD_TYPE=DEBUG
# mpirun -n 2 xterm -hold -e gdb -ex run --args ./test_mat 10 7
# mpirun -n 2 xterm -hold -e ./test_mat 10 7
# mpirun  --oversubscribe -n 4 konsole --hide-menubar --hold -e ./test_mat 30 15