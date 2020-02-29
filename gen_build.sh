#!/bin/bash
mkdir build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=RELEASE;
make test_mat;