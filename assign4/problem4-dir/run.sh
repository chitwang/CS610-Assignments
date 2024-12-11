nvcc -ccbin /usr/bin/g++-10 -O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx problem4.cu -o problem4.out
echo ""
echo "Executing..."
./problem4.out
