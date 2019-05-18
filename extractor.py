import os
import pdb

import numpy as np
import shutil

flips = [2] # ,2,4] #,8]
entropy = [0, 2, 4, 8, 16,16,16]  #16, 100] # 100 is dummy
PATH = 'output'

flag= False
for fl in flips :
    flip_path = os.path.join(PATH, 'change_{}'.format(fl))
    save_path = os.path.join(PATH, 'extracted_{}'.format(fl))
    if not os.path.exists(save_path) :
        os.mkdir(save_path)
    trials = sorted(os.listdir(flip_path))
    pivot = 4
    st = 120
    for c,t in enumerate(trials) :
        trial_path = os.path.join(flip_path, t)
        matrices = sorted(os.listdir(trial_path))
        diff = 1000

        if (not '_{}_'.format(st) in t) : #Eand (not flag):
            flag = True 
            continue 
        qw = c
        st += 1
        for matrix in matrices :
            if 'zmask' in matrix :
                continue 
            elif 'n_' in matrix or 'p_' in matrix :
                ent = int(matrix.split('_')[2])
                if abs(entropy[pivot]*10 - ent) < diff :
                    fit_mat = matrix
                    diff = abs(entropy[pivot]*10 - ent)
                    
            else :
                mat_path = os.path.join(trial_path, matrix)
                mov_path = os.path.join(save_path, matrix)
        if diff > 1 :
            print(diff)
            continue
        matrix_path = os.path.join(trial_path, fit_mat)
        move_path = os.path.join(save_path, fit_mat)
        shutil.copy2(matrix_path, move_path)
        shutil.copy2(mat_path, mov_path)
        print("MOVE " + fit_mat)

        if (c+1)%30 == 0 :
            pivot += 1 
