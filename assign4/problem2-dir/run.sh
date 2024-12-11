nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx problem2.cu -o problem2.out
echo ""
echo "Executing..."
./problem2.out
