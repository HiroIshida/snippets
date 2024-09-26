nasm -f elf64 -o memory.o memory.s
g++ main.cpp memory.o
