for f in "-mno-avx -mfma BENCH_SSE" "-mavx -mavx2 -mfma BENCH_AVX" "-mavx512f -mfma BENCH_AVX512"; do
    g++ -O3 -w quat_vec_mult.cpp -I/usr/include/eigen3 -std=c++20 ${f% *}
    echo "${f##* }"
    ./a.out
done
