#!/bin/bash
for fl in 1 2 4 8
do 
 echo python3 main.py --num_flips $fl --shape 8 --num_iters 1000 --num_matrices 150 --save_path 'output'
 python3 main.py --num_flips $fl --shape 8 --num_iters 1000 --num_matrices 150 --save_path 'output'
done
