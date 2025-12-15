#!/bin/bash

echo "Operating system: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
echo "Kernel: $(uname -s) $(uname -r)"
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -n1 | cut -d':' -f2 | xargs) ($(grep -c processor /proc/cpuinfo) threads available)"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) | Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
else
    echo "GPU: Not found or nvidia-smi missing"
fi

echo "Compiler (GCC): $(gcc --version | head -n1)"

if command -v nvcc &> /dev/null; then
    echo "Compiler (NVCC/CUDA): $(nvcc --version | grep release | awk '{print $5, $6}')"
else
    echo "Compiler (NVCC/CUDA): Not found"
fi

omp_raw=$(echo | cpp -fopenmp -dM 2>/dev/null | grep "_OPENMP" | awk '{print $3}')
case "$omp_raw" in
    202411) omp_ver="6.0" ;;
    202111) omp_ver="5.2" ;;
    202011) omp_ver="5.1" ;;
    201811) omp_ver="5.0" ;;
    201511) omp_ver="4.5" ;;
    201307) omp_ver="4.0" ;;
    201107) omp_ver="3.1" ;;
    200805) omp_ver="3.0" ;;
    200505) omp_ver="2.5" ;;
    200203) omp_ver="2.0" ;;
    199810) omp_ver="1.0" ;;
    *)      omp_ver="Unknown ($omp_raw)" ;;
esac
echo "OpenMP: Version $omp_ver"