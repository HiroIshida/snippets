n_iter = 100000

asm_string = "section .text\n"

# define movsd_xmm_xmm_bench
function_name = "movsd_xmm_xmm_bench"
asm_string += f"global {function_name}\n"
asm_string += "movsd_xmm_xmm_bench:\n"
for _ in range(n_iter):
    asm_string += "movsd xmm0, xmm1\n"
    asm_string += "movsd xmm1, xmm0\n"
asm_string += "ret\n"
asm_string += "\n"

# define movsd_xmm_mem_bench
function_name = "movsd_xmm_stack_bench"
asm_string += f"global {function_name}\n"
asm_string += "movsd_xmm_stack_bench:\n"

asm_string += "push rbp\n"
asm_string += "mov rbp, rsp\n"
asm_string += "sub rsp, 8\n"
for _ in range(n_iter):
    asm_string += "movsd xmm0, [rsp]\n"
    asm_string += "movsd [rsp], xmm0\n"
asm_string += "add rsp, 8\n"
asm_string += "pop rbp\n"
asm_string += "ret\n"

with open("bench.s", "w") as f:
    f.write(asm_string)
