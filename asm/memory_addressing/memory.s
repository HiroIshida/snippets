default rel
section .rdata
PrimesNums: dd 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

section .text
global access_rdata
;; int32_t access_rdata(int32_t index);
;; rdi, rsi, rdx, rcx, r8, r9
;; edi, esi, edx, ecx, r8d, r9d
access_rdata:
movsxd r10, edi
shl r10, 2
lea r11, [PrimesNums]
add r11, r10
mov eax, [r11]
ret
