#!/bin/bash

for k in 1 2 ; do echo "k=$k a.txt ---------------"   ; mpirun -np $k ./test_mat 4 3 /data/bgch_files/matr/a.txt   ; done
for k in 1 2 ; do echo "k=$k a20.txt ---------------" ; mpirun -np $k ./test_mat 4 3 /data/bgch_files/matr/a20.txt ; done
for k in 1 2 ; do echo "k=$k b.txt ---------------"   ; mpirun -np $k ./test_mat 4 3 /data/bgch_files/matr/b.txt   ; done
for k in 1 2 ; do echo "k=$k c.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 /data/bgch_files/matr/c.txt   ; done
for k in 1 2 ; do echo "k=$k d.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 /data/bgch_files/matr/d.txt   ; done
for k in 1 2 ; do echo "k=$k e.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 /data/bgch_files/matr/e.txt   ; done
for k in 1 2 ; do echo "k=$k f.txt ---------------"   ; mpirun -np $k ./test_mat 6 3 /data/bgch_files/matr/f.txt   ; done
