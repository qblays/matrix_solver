# matrix_solver
Solves symmetric matrices using Cholesky decomposition.
# Features
- MPI support
- AVX support
# Build from sources
```
git clone https://github.com/qblays/matrix_solver
mkdir build; cd build  # Create a build directory.
cmake ..               # Configure the project.
cmake --build .        # Build all default targets.
```

# Examples
```
mpirun -np 2 xterm -hold -e gdb -ex run --args ./test_mat 10 7 # Debug run
mpirun -np 2 xterm -hold -e ./test_mat 10 7
mpirun -np 4 konsole --hide-menubar --hold -e ./test_mat 30 15
```