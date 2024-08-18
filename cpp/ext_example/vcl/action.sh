# g++ bench_quaternion.cpp -march=native -O2 -g -o bench_quaternion
g++ bench_mat.cpp -I/usr/include/eigen3 -march=native -O3 -g -o bench_mat
