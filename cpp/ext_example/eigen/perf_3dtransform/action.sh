#!/bin/bash
option_list=(
    "-O2"
    "-O2 -mavx -mavx2"
    "-O2 -march=native"
)
for option in "${option_list[@]}"; do
    echo "bench ${option}"
    g++ main.cpp -I/usr/include/eigen3 ${option} -o main
    ./main
done
