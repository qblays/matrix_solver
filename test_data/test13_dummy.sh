#!/bin/bash
# 3000 runs
for ((n=3; n<=30;n++)) ; do for ((m=3;m<=$n;m+=3)) ; do for ((k=1;k<=$n;k++)) ; do echo "n=$n m=$m k=$k ----------------" ; done ; done ; done
