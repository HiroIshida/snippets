n_iter = 100000

def create_math_bench(instr_name):
    # define addsd_bench
    asm_string = ""
    function_name = f"{instr_name}_operation_bench"
    asm_string += f"global {function_name}\n"
    asm_string += f"{function_name}:\n"
    for _ in range(n_iter):
        asm_string += f"{instr_name} xmm0, xmm1\n"
        asm_string += f"{instr_name} xmm0, xmm1\n"
    asm_string += "ret\n"
    asm_string += "\n"
    return asm_string


def create_xmm_xmm_bench(instr_name):
    # define movsd_xmm_xmm_bench
    asm_string = ""
    function_name = f"{instr_name}_xmm_xmm_bench"
    asm_string += f"global {function_name}\n"
    asm_string += f"{function_name}:\n"
    for _ in range(n_iter):
        asm_string += f"{instr_name} xmm0, xmm1\n"
        asm_string += f"{instr_name} xmm1, xmm0\n"
    asm_string += "ret\n"
    asm_string += "\n"
    return asm_string


def create_xmm_rax_bench(instr_name):
    # define movsd_xmm_xmm_bench
    asm_string = ""
    function_name = f"{instr_name}_xmm_rax_bench"
    asm_string += f"global {function_name}\n"
    asm_string += f"{function_name}:\n"
    for _ in range(n_iter):
        asm_string += f"{instr_name} xmm0, rax\n"
        asm_string += f"{instr_name} rax, xmm0\n"
    asm_string += "ret\n"
    asm_string += "\n"
    return asm_string


def create_xmm_stack_bench(instr_name):
    # define movsd_xmm_mem_bench
    asm_string = ""
    function_name = f"{instr_name}_xmm_stack_bench"
    asm_string += f"global {function_name}\n"
    asm_string += f"{function_name}:\n"

    asm_string += "push rbp\n"
    asm_string += "mov rbp, rsp\n"
    asm_string += "sub rsp, 16\n"  # 16 bytes for xmm alignment
    for _ in range(n_iter):
        asm_string += f"{instr_name} xmm0, [rsp]\n"
        asm_string += f"{instr_name} [rsp], xmm0\n"
    asm_string += "add rsp, 16\n"
    asm_string += "pop rbp\n"
    asm_string += "ret\n"
    asm_string += "\n"
    return asm_string

asm_string = "section .text\n"
asm_string += create_xmm_xmm_bench("movsd")
asm_string += create_xmm_stack_bench("movsd")
asm_string += create_xmm_xmm_bench("movapd")
asm_string += create_xmm_stack_bench("movapd")

asm_string += create_xmm_rax_bench("movq")
asm_string += create_xmm_rax_bench("vmovq")


asm_string += create_math_bench("addsd")
asm_string += create_math_bench("subsd")
asm_string += create_math_bench("mulsd")

with open("bench.s", "w") as f:
    f.write(asm_string)
