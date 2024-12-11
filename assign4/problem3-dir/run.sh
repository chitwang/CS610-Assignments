echo "Compiling the files"
gcc -O2 -std=c17 -D_POSIX_C_SOURCE=199309L problem3-v0.c -o problem3-v0.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem3-a.cu -o problem3-a.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem3-b.cu -o problem3-b.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem3-c.cu -o problem3-c.out
nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_61 -lineinfo -res-usage -src-in-ptx problem3-d.cu -o problem3-d.out

echo "Running the executables"
echo "Serialized C Code"
./problem3-v0.out
echo ""
echo "Naive CUDA Kernel"
./problem3-a.out
echo ""
echo "UVM + Optimisations CUDA Kernel"
./problem3-b.out
echo ""
echo "UVM CUDA Kernel"
./problem3-c.out
echo ""
echo "Thrust Kernel"
./problem3-d.out

echo "Running diff over versions"

diff results-v0.txt results-va.txt
if [ $? -eq 0 ]; then
    echo "results-v0.txt and results-va.txt are consistent."
else
    echo "results-v0.txt and results-va.txt are NOT consistent."
fi

diff results-v0.txt results-vb.txt
if [ $? -eq 0 ]; then
    echo "results-v0.txt and results-vb.txt are consistent."
else
    echo "results-v0.txt and results-vb.txt are NOT consistent."
fi

diff results-v0.txt results-vc.txt
if [ $? -eq 0 ]; then
    echo "results-v0.txt and results-vc.txt are consistent."
else
    echo "results-v0.txt and results-vc.txt are NOT consistent."
fi

diff results-v0.txt results-vd.txt
if [ $? -eq 0 ]; then
    echo "results-v0.txt and results-vd.txt are consistent."
else
    echo "results-v0.txt and results-vd.txt are NOT consistent."
fi

echo "All diff checks completed."
