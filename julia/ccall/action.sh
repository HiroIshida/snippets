export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hiro/documents/misc/snippets/julia/ccall
gcc -shared -fPIC -o libmylib.so test.c 
