"""
Python implementation of the matrix information measurement examples from the
StackExchange answer written by WilliamAHuber for
"Measuring entropy/ information/ patterns of a 2d binary matrix"
http://stats.stackexchange.com/a/17556/43909

Copyright 2014 Cosmo Harrigan
This program is free software, distributed under the terms of the GNU LGPL v3.0

Modified by Jaedong Hwang for experiments of Cognitive Psychology
"""

#__author__ = 'Cosmo Harrigan'
__author__ = 'Jaedong Hwang'

from matplotlib import pyplot
from neighborhood_functions import avg_components
from moving_window_filter import moving_window_filter
from calculate_profile import profile
import numpy as np 
import os
# Function to apply
F = avg_components
import random

import cv2

# Define the matrices as input_matrices
import pdb
import argparse
from PIL import Image

# Iterate over the input matrics

PATH='output/'

def save_matrix(idx,shape, score,matrix, size=512,name='mat',  path='output',ratio=None) :
    if not os.path.exists(path) :
        os.mkdir(path)
    img = Image.fromarray(np.uint8(matrix*255))
    img2 = cv2.resize(np.uint8(matrix), (size,size), interpolation=cv2.INTER_NEAREST)
    #Image.fromarray(np.uint8(img2*255)).save(os.path.join(path, '{:02d}_{:04d}.png'.format(shape,idx)))
    #Image.fromarray(np.uint8(img2*255)).save(os.path.join(path, '{}{:04d}_{:04}.png'.format(name, idx,int(score*1000))))
    if ratio is None :
        Image.fromarray(np.uint8(img2*255)).save(os.path.join(path, '{}{:04d}.png'.format(name, int(round(score*100)))))
    else :
        if ratio < 0 :
            Image.fromarray(np.uint8(img2*255)).save(os.path.join(path, '{}{:04d}_n_{:}.png'.format(name, int(round(-ratio*1000)), int(round(score*100)))))
        else :
            Image.fromarray(np.uint8(img2*255)).save(os.path.join(path, '{}{:04d}_p_{:}.png'.format(name, int(round(ratio*1000)), int(round(score*100)))))


def calculate_entropy(matrix) :
    matrices = []
    # Produce the filtered matrices at varying scales and the associated
    # entropy "profiles"
    for n in range(1, min(matrix.shape)):
        output_matrix = moving_window_filter(matrix=matrix,
                                             f=F,
                                             neighborhood_size=n)
        matrices.append(output_matrix)
#            subplot = pyplot.subplot(num_iters, 7, m * 7 + n) # row col index
#            pyplot.axis('off')
#            pyplot.imshow(output_matrix,
#                          interpolation='nearest',
#                          cmap='Greys_r',
#                          vmin=0,
#                          vmax=1)
#        print("Neighborhood size = {0}\n{1}\n".format(n, output_matrix))
#   pyplot.show()
    prof = np.array(profile(matrices))
    return prof.mean(), prof, matrices


import operator as op
from functools import reduce

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def flipper (shape, num_flips, used={}, seed=0) :
    C = comb(shape**2, num_flips)
    while True :
        if len(used) == C: 
            return None,None,seed
        np.random.seed(seed)
        perm = np.random.permutation(shape**2)
        perm = np.sort(perm[:num_flips])
        key = np.array2string(perm,separator=',')
        seed += 1
        if key not in used :
            break
    used[key] = True
    return perm, used, seed
    checker = np.zeros((shape, shape))
    # random permutation
    # num_flips cut 
    # sorted
    # check whether it is used or not.
    # return
    return checker, permuted



def parse_args() :
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--num_matrices', dest='num_matrices', default=30, type=int)
    parser.add_argument('--num_iters', dest='num_iters', default=100, type=int)
    parser.add_argument('--num_flipss', dest='num_flips', default=4, type=int)
    parser.add_argument('--shape', dest='shape', default=8, type=int)
    parser.add_argument('--save_path', dest='save_path', default='output', type=str)
    args = parser.parse_args()
    return args

def mat2key(mat) :
    fl = mat.reshape(-1) 
    key = ''.join(str(e) for e in fl)
    return key

if __name__ == '__main__':
    args = parse_args()
    print(args)
    num_matrices = args.num_matrices #10 # 20
    shape = args.shape # 8
    num_iters = args.num_iters # 1000
    num_flips = args.num_flips# 8
    PATH = os.path.join(args.save_path, 'change_'+str(num_flips))
    tr = 0
    if not os.path.exists(PATH) :
        os.mkdir(PATH)

    list_dir = os.listdir(PATH)
    for e in range(len(list_dir)) :
        if 'trial_{:03d}'.format(e) in list_dir:
            tr = e
    used_originals = {}
    for m in range(num_matrices):
        seed = random.randint(0,2147483647)
        #matrix = input_matrices[m]
        score = -1 
        key = 'KEY'
        while abs(score - 2) > 0.02 and (not key in used_originals):
            matrix = np.random.rand(shape,shape)
            #matrix = np.random.rand(shape,shape)
            matrix = (matrix > 0.5)
            score, _, _ = calculate_entropy(matrix)
            seed += 1
            key = mat2key(matrix)

        path = os.path.join(PATH,'trial_{:03d}_{:03d}_{}'.format(tr+m,int(round(score*100)), seed))
        save_matrix(m,shape, score, matrix,name='aoriginal', path=path)
        # TODO make flipper
        used = {}
        flattened_matrix = matrix.flatten()
        for i in range(num_iters):
            checker = np.zeros(shape**2)
            perm, used, seed = flipper(shape, num_flips, used, seed)
            if perm is None :
                print("Already used all combination!")
                break
            checker[perm] = 1
            checker = checker.reshape((shape,shape)).astype(np.bool)
            new_matrix = (checker^matrix)
#            flattened_matrix[perm] = 1 - flattened_matrix[perm] # flipped
#            new_matrix = flattened_matrix.reshape((shape,shape))
            new_score, _, _ = calculate_entropy(new_matrix)
            ratio = (new_score - score) / score
            if i % 100 ==0 : 
                print("{:01d} {:.2f} {:.2f} {:.3f}".format(i, score, new_score, ratio))
            save_matrix(2, shape, new_score, new_matrix^matrix,name='zmask',  path=path, ratio=ratio)
            save_matrix(1, shape, new_score,new_matrix, path=path, ratio=ratio)
        if m % 10 == 0 :
            print("{}".format(m))
    print("DONE")
