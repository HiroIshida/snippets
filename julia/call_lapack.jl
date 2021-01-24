const liblapack = Base.liblapack_name
import LinearAlgebra.LAPACK: chklapackerror
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra: BlasFloat, BlasInt, LAPACKException, I, cholesky
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare,

using LinearAlgebra: triu, tril, dot

using Base: iszero, require_one_based_indexing

function dpbtf!(uplo::AbstractChar, n::Integer, kd::Integer, AB::AbstractMatrix{Float64})
    #chkuplo(uplo)
    LDAB = max(1,stride(AB,2))
    info = Ref{BlasInt}()
    println("calling dpbtrf_")
    ccall((@blasfunc(dpbtrf_), liblapack), Cvoid,
          (Ref{UInt8},
           Ref{BlasInt},
           Ref{BlasInt},
           Ptr{Float64},
           Ref{BlasInt},
           Ptr{BlasInt}),
          uplo, n, kd, AB, LDAB, info)
    chklapackerror(info[])
    return AB
end
# as for doc for dpbtrf
# http://www.netlib.org/lapack/lapack-3.1.1/html/dpbtrf.f.html
# as for strage of banded symetric matrix
# https://www.netlib.org/lapack/lug/node124.html
#
# Fro cholesky to be done, A must be positive definite. So, it is strange that Toussaint's paper claims that they can use dpbtf to the least square norm which is positive semi-defintie.
#AB = [1+0.01 -1; 2+0.01 -1; 1+0.01 0.]
AB = [1.01 2.01 1.01; -1. -1 0]
uplo = 'L'
out = dpbtf!(uplo, 3, 1, AB)

A = [1 -1 0.;
     -1 2 -1;
     0 -1 1]
lambda = 0.01
cholesky(A+lambda * Matrix{Float64}(I, 3, 3))





for (gbtrf, gbtrs, elty) in
    ((:dgbtrf_,:dgbtrs_,:Float64),
     (:sgbtrf_,:sgbtrs_,:Float32),
     (:zgbtrf_,:zgbtrs_,:ComplexF64),
     (:cgbtrf_,:cgbtrs_,:ComplexF32))
    @eval begin
        # SUBROUTINE DGBTRF( M, N, KL, KU, AB, LDAB, IPIV, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, KL, KU, LDAB, M, N
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * )
        function gbtrf!(kl::Integer, ku::Integer, m::Integer, AB::AbstractMatrix{$elty})
            require_one_based_indexing(AB)
            chkstride1(AB)
            n    = size(AB, 2)
            mnmn = min(m, n)
            ipiv = similar(AB, BlasInt, mnmn)
            info = Ref{BlasInt}()
            ccall((@blasfunc($gbtrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, kl, ku, AB, max(1,stride(AB,2)), ipiv, info)
            chklapackerror(info[])
            AB, ipiv
        end

        # SUBROUTINE DGBTRS( TRANS, N, KL, KU, NRHS, AB, LDAB, IPIV, B, LDB, INFO)
        # *     .. Scalar Arguments ..
        #       CHARACTER          TRANS
        #       INTEGER            INFO, KL, KU, LDAB, LDB, N, NRHS
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )
        function gbtrs!(trans::AbstractChar, kl::Integer, ku::Integer, m::Integer,
                        AB::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt},
                        B::AbstractVecOrMat{$elty})
            require_one_based_indexing(AB, B)
            chkstride1(AB, B, ipiv)
            chktrans(trans)
            info = Ref{BlasInt}()
            n    = size(AB,2)
            if m != n || m != size(B,1)
                throw(DimensionMismatch("matrix AB has dimensions $(size(AB)), but right hand side matrix B has dimensions $(size(B))"))
            end
            ccall((@blasfunc($gbtrs), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Clong),
                  trans, n, kl, ku, size(B,2), AB, max(1,stride(AB,2)), ipiv,
                  B, max(1,stride(B,2)), info, 1)
            chklapackerror(info[])
            B
        end
    end
end
