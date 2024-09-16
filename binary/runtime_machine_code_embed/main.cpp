#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <cstring>

int main() {
    size_t page_size = getpagesize();
    std::cout << "page size: " << page_size << std::endl;
    void *mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        std::cerr << "mmap failed" << std::endl;
        return 1;
    }

    uint8_t *instruction = static_cast<uint8_t *>(mem);

    size_t n_repeat;
    std::cout << "Enter the number of times to repeat the function: ";
    std::cin >> n_repeat;

    uint8_t *func_code = new uint8_t[n_repeat * 4 + 1]; 
    for (size_t i = 0; i < n_repeat; ++i) {
        func_code[i * 4 + 0] = 0xF2;
        func_code[i * 4 + 1] = 0x0F;
        func_code[i * 4 + 2] = 0x58;
        func_code[i * 4 + 3] = 0xC1;
    }
    func_code[n_repeat * 4] = 0xC3;

    std::memcpy(instruction, func_code, n_repeat * 4 + 1);
    if (mprotect(mem, page_size, PROT_READ | PROT_EXEC) == -1) {
        delete[] func_code;
        return 1;
    }

    using Func = double(*)(double, double);
    Func add_func = reinterpret_cast<Func>(instruction);

    double x = 1.7;
    double y = 2.5;
    double result = add_func(x, y);

    std::cout << x << " + " << y << " * " << n_repeat << " = " << result << std::endl;

    delete[] func_code;
    munmap(mem, page_size);

    return 0;
}
