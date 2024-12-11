#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <TBB_FLAG> <ops> <rns> <add> <rem>"
    echo "Provide 1 or 0 as the first argument (TBB_FLAG)."
    exit 1
fi

# Read arguments
flag=$1
ops=$2
rns=$3
add=$4
rem=$5

# Check the value of the argument
if [ "$1" == "1" ]; then
    echo "TBB Flag is 1"
     echo "Compiling..."
      g++ -O3 -std=c++17 -fopenmp -DUSE_TBB problem1.cpp -o problem1.out -ltbb
     echo "Executing ./problem1.out -ops=$ops -rns=$rns -add=$add -rem=$rem"
	./problem1.out -ops=$ops -rns=$rns -add=$add -rem=$rem
elif [ "$1" == "0" ]; then
    echo "TBB Flag is 0"
    echo "Compiling..."
    g++ -O3 -std=c++17 -fopenmp problem1.cpp -o problem1.out
    echo "Executing ./problem1.out -ops=$ops -rns=$rns -add=$add -rem=$rem"
    ./problem1.out -ops=$ops -rns=$rns -add=$add -rem=$rem 
else
    echo "Invalid flag. Please use 1 or 0."
    exit 1
fi

