n_repeat = 100
asmcode = "section .text\n"

# benchmark for naive spill
asmcode += "global bench1\n"
asmcode += "bench1:\n"
asmcode += "push rbp\n"
asmcode += "mov r12, rdi\n"
asmcode += "mov rbp, rsp\n"
asmcode += "sub rsp, 1024\n"  # whatever
for _ in range(n_repeat):
    for i in range(16):
        asmcode += "vmovsd [rsp + {}], xmm{}\n".format(8 * i, i)
    asmcode += "call r12\n"
asmcode += "mov rsp, rbp\n"
asmcode += "pop rbp\n"
asmcode += "ret\n"

# benchmark for simd spill (use unpcklpd)
asmcode += "global bench2\n"
asmcode += "bench2:\n"
asmcode += "push rbp\n"
asmcode += "mov r12, rdi\n"
asmcode += "mov rbp, rsp\n"
asmcode += "and rsp, -16\n"  # align stack to 16 bytes
asmcode += "sub rsp, 1024\n"  # whatever
for _ in range(n_repeat):
    for i in range(8):
        asmcode += "vunpcklpd xmm{}, xmm{}, xmm{}\n".format(2 * i, 2 * i, 2 * i + 1)
    for i in range(8):
        asmcode += "vmovapd [rsp + {}], xmm{}\n".format(16 * i, 2 * i)
    asmcode += "call r12\n"
asmcode += "mov rsp, rbp\n"
asmcode += "pop rbp\n"
asmcode += "ret\n"

# bemchmark for simd spill (use  movaps shufpd)
asmcode += "global bench3\n"
asmcode += "bench3:\n"
asmcode += "push rbp\n"
asmcode += "mov r12, rdi\n"
asmcode += "mov rbp, rsp\n"
asmcode += "and rsp, -16\n"  # align stack to 16 bytes
asmcode += "sub rsp, 1024\n"  # whatever
for _ in range(n_repeat):
    for i in range(8):
        asmcode += "shufpd xmm{}, xmm{}, 0\n".format(2 * i, 2 * i + 1)
    for i in range(8):
        asmcode += "vmovapd [rsp + {}], xmm{}\n".format(16 * i, 2 * i)
    asmcode += "call r12\n"
asmcode += "mov rsp, rbp\n"
asmcode += "pop rbp\n"
asmcode += "ret\n"

with open("bench.s", "w") as f:
    f.write(asmcode)
