import argparse
import pandas as pd
import numpy as np
import math
from hmmlearn import hmm
import random

import os
from os.path import isfile, join


###parser part
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="name of file of data to be train of")
parser.add_argument("--test", help="name of folder of tests files")
parser.add_argument("--output", help="name of file to be resault be saved")
args = parser.parse_args()


if args.train:
    TRAIN_NAME = args.train
else:
    TRAIN_NAME = "house3_5devices_train.csv"

if args.test:
    TEST_NAME = args.test
else:
    TEST_NAME = "test_folder"

if args.output:
    RESULT_NAME = args.output
else:
    RESULT_NAME = "results.txt"


#reading data
tests = []
for filename in os.listdir(TEST_NAME):
    df = pd.read_csv(os.path.join(os.getcwd(),TEST_NAME ,filename))
    df = df.iloc[:, 1].values
    tests.append(df)
nr_of_tests = len(tests)

train = pd.read_csv(TRAIN_NAME).iloc[:,1:]

train_ligthing2 = train.iloc[:,0].values.reshape(-1, 1)
train_ligthing5 = train.iloc[:,1].values.reshape(-1, 1)
train_ligthing4 = train.iloc[:,2].values.reshape(-1, 1)
train_refrigerator = train.iloc[:,3].values.reshape(-1, 1)
train_microwave = train.iloc[:,4].values.reshape(-1, 1)


##
def modelowanie(n, X):
    X_train = X[:X.shape[0] // 2]
    X_validate = X[X.shape[0] // 2:]
    best_score = best_model = None
    best_aic = best_model_aic = None
    for idx in range(1, n):
        model = hmm.GaussianHMM(n_components=idx)  # czy jeszcze jakieś inne parametry?
        model.fit(X_train)
        score = model.score(X_validate)
        print(f'Model #{idx}\tScore: {score}')
        if best_score is None or score > best_score:
            best_model = model
            best_score = score
        AIC = model.aic(X_validate)
        print(f'Model AIC #{idx}\tScore: {AIC}')
        if best_aic is None or AIC < best_aic:
            best_model_aic = model
            best_aic = AIC

    print(f'Best score:      {best_score}')
    print(f'Best AIC:      {best_aic}')


modelowanie(11, train_microwave)
#### cuting the data


#k - dlugosc danych, N - ile powtorzen col - ktora kolumna(urzadzenie)
def stworz_nowe(train,k = 1000,col = 0,N = 100):
    cutted = train.iloc[:,col].values
    n = len(cutted)
    od_kad = np.random.randint(low = 0, high=n-k-1, size=N)

    return None




















































