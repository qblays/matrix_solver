#!/bin/bash
dir="${BASH_SOURCE%/*}"
for k in 1 2 ; do echo "k=$k a.txt ---------------"   ; mpirun -np $k ./test_mat 4 3 ${dir}/a.txt   ; done
for k in 1 2 ; do echo "k=$k a20.txt ---------------" ; mpirun -np $k ./test_mat 4 3 ${dir}/a20.txt ; done
for k in 1 2 ; do echo "k=$k b.txt ---------------"   ; mpirun -np $k ./test_mat 4 3 ${dir}/b.txt   ; done
for k in 1 2 ; do echo "k=$k c.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 ${dir}/c.txt   ; done
for k in 1 2 ; do echo "k=$k d.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 ${dir}/d.txt   ; done
for k in 1 2 ; do echo "k=$k e.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 ${dir}/e.txt   ; done
for k in 1 2 ; do echo "k=$k f.txt ---------------"   ; mpirun -np $k ./test_mat 4 3 ${dir}/f.txt   ; done
