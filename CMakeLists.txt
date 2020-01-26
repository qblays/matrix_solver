cmake_minimum_required(VERSION 3.0)
find_package(Threads REQUIRED)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)


#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
    #message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
#file(GLOB_RECURSE  sources "src/*.cpp" )
#file(GLOB_RECURSE  test_mpi_src "test/test_mpi.cpp" )
set(test_mpi_src "test/test_mpi.cpp" )
set(test_mat_src "test/test_mat.cpp" )
foreach (_srcName ${test_mpi_src})
    message(STATUS "${_srcName}")
endforeach()
# foreach (_srcName ${tests})
#     message(STATUS "${_srcName}")
# endforeach()

#add_executable(prog ${sources})
#set(CXX_FLAGS "-O3 --fast-math -Wall -march=native")
set(CXX_FLAGS "-O0 -Wall -g")
#add_compile_options(-O3 --fast-math -Wall)
add_compile_options(-O0 -g -Wall)
add_executable(test_mpi ${test_mpi_src})
add_executable(test_mat ${test_mat_src})
# mpirun -n 2 xterm -hold -e gdb -ex run --args ./test_mat 10 7