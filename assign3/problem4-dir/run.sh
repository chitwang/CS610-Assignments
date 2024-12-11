gcc -O3 -std=c17 -D_POSIX_C_SOURCE=199309L  problem4-v0.c -o problem4-v0.out -lm
gcc -g -std=c17 -masm=att -msse4 -mavx2 -march=native -O3 -fopenmp -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions  210295-prob4-v1.c -o 210295-prob4-v1.out
gcc -g -std=c17 -masm=att -msse4 -mavx2 -march=native -O3 -fopenmp -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions  210295-prob4-v2.c -o 210295-prob4-v2.out
