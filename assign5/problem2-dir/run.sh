#!/bin/bash

echo "Compiling the code..."
g++ -std=c++17 -O3 problem2.cpp -o problem2.out -latomic
echo ""
# Iterate NUM_OPS from 1e5 to 1e7
for ops in 100000 500000 1000000 5000000 10000000; do
    # Iterate NUM_THREADS from 1 to 16 in powers of 2
    for threads in 1 2 4 8 16; do
        echo "Running with NUM_OPS=$ops and NUM_THREADS=$threads"
        ./problem2.out $ops $threads
        echo "Completed for NUM_OPS=$ops and NUM_THREADS=$threads"
        echo "------------------------------------"
    done
done
