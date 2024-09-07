default rel
section .text
global AddSubI32
AddSubI32:
    add edi,esi
    add edx,ecx
    sub edi,edx
    add edi,7
    mov eax,edi
    ret

