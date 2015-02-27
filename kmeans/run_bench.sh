#!/bin/bash



for data in ~/645/data/*.dat
do
    ./cuda_main -i $data -n 32 -o
    sleep 1
done
