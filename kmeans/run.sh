#/bin/bash

make omp_main
./omp_main -i kmeans01.dat -n 3 -o -a
./omp_main -i kmeans02.dat -n 3 -o -a
./omp_main -i kmeans03.dat -n 3 -o -a
./omp_main -i kmeans04.dat -n 3 -o -a

make cuda_main
./cuda_main -i kmeans01.dat -n 3 -o
./cuda_main -i kmeans02.dat -n 3 -o
./cuda_main -i kmeans03.dat -n 3 -o
./cuda_main -i kmeans04.dat -n 3 -o
