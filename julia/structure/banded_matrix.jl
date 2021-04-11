using LinearAlgebra

# special packing strategy
#https://www.netlib.org/lapack/lug/node124.html

AB = [1 0; 2 2;1 0; 2 2; 1 0.]
# means that [1, 2, 0;
#             2, 1, 1;
#             1, 2, 1]
LinearAlgebra.LAPACK.gbtrf!(1, 1, 3, AB)

# note also that symetric matrix is packed in a different way in lapack
