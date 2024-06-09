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

train_ligthing2 = train.iloc[:,0].values
train_ligthing5 = train.iloc[:,1].values
train_ligthing4 = train.iloc[:,2].values
train_refrigerator = train.iloc[:,3].values
train_microwave = train.iloc[:,4].to_numpy
##
def modelowanie(n, X):
    model = hmm.GaussianHMM(n_components=n).fit(X) #czy jeszcze jakieś inne parametry?
    hidden_states = model.predict(X)
    print('done')

    return hidden_states

modelowanie(2,train_microwave)
#### cuting the data


#k - dlugosc danych, N - ile powtorzen col - ktora kolumna(urzadzenie)
def stworz_nowe(train,k = 1000,col = 0,N = 100):
    cutted = train.iloc[:,col].values
    n = len(cutted)
    np.random.randint(low = 0, high=n-k-1, size=None, dtype=int)

    return None




















































