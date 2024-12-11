echo "Compiling the files"
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem1-a.cu -o problem1-a.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem1-b.cu -o problem1-b.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem1-c.cu -o problem1-c.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem1-d.cu -o problem1-d.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem1-e.cu -o problem1-e.out

echo "Running the executables"

echo "Naive CUDA Kernel"
./problem1-a.out
echo ""
echo "Shared memory Kernel"
./problem1-b.out
echo ""
echo "Shared memory with Loop Transformations"
./problem1-c.out
echo ""
echo "Shared memory with Loop Transformations and Pinned Memory"
./problem1-d.out
echo ""
echo "Shared memory with Loop Transformations and UVM"
./problem1-e.out
echo ""