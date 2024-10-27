python3 asmgen.py
nasm -f elf64 -o bench.o bench.s
g++ main.cpp bench.o -o main
