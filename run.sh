#!/bin/bash 

n=409600000

ml list


make clean
make MODEL=""
echo "GPU Acccessing HBM"
./stream -n $n


make clean
make MODEL="-DHOST"
echo "GPU Acccessing managed host allocated memory"
./stream -n $n

make clean
make MODEL="-DZERO_COPY"
echo "GPU Acccessing LPDDR5 with no page migration"
./stream -n $n
