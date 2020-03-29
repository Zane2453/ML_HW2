import argparse

import numpy as np
from math import factorial

def Combinations(n,r):
    return factorial(n) / factorial(r) / factorial(n-r)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', dest='win', type=int, default=0, help="initial beta prior for win")
    parser.add_argument('-b', dest='lose', type=int, default=0, help="initial beta prior for lose")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()
    prior = np.array([args.win, args.lose])

    testfile = open('./testfile.txt', 'r')
    for index, line in enumerate(testfile):
        win = 0
        lose = 0
        for text in line.strip():
            if text == '1':
                win += 1
            elif text == '0':
                lose += 1
        prob = win / (win + lose)
        likelihood = Combinations(win+lose, win) * (prob ** win) * ((1 - prob) ** lose)

        print(f'case {index+1}: {line.strip()}')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior:     a = {prior[0]} b = {prior[1]}')
        prior += np.array([win, lose])
        print(f'Beta posterior: a = {prior[0]} b = {prior[1]}\n')




