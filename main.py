import argparse
import pandas as pd
import numpy as np
import math
from hmmlearn import hmm
import random
import itertools

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
def modelowanie_log(n, X):
    X = X[4001:]
    X_train = X[:int(0.9*len(X))]
    X_validate = X[int(0.9*len(X)):]
    models = {}
    for idx in range(1, n):
        model = hmm.GaussianHMM(n_components=idx)  # czy jeszcze jakieś inne parametry?
        model.fit(X_train)
        score = model.score(X_validate)
        models[score]= model
    models = dict(sorted(models.items()))
    values = list(models.values())
    best_three_values = values[-3:]
    return(best_three_values)




def modelowanie_AIC(n, X):
    X = X[4001:]
    X_train = X[:int(0.9 * len(X))]
    X_validate = X[int(0.9 * len(X)):]
    models = {}
    for idx in range(1, n):
        model = hmm.GaussianHMM(n_components=idx)  # czy jeszcze jakieś inne parametry?
        model.fit(X_train)
        Aic = model.aic(X_validate)
        models[Aic] = model
    models = dict(sorted(models.items()))
    values = list(models.values())
    best_three_values = values[1:3]
    return (best_three_values)


#modelowanie_log(11, train_microwave)
#### cuting the data


#k - dlugosc danych, N - ile powtorzen col - ktora kolumna(urzadzenie)
def stworz_nowe(train,k = 1000,col = 0,N = 100):
    cutted = train.iloc[:,col].values
    n = len(cutted)
    od_kad = np.random.randint(low = 0, high=n-k-1, size=N)

    return None

a=[0,1,2]

listy = list(itertools.product(a,a,a,a,a))
do_mocy = {}
for lis in listy:
    do_mocy[tuple(lis)] = 0

n = 11
dobre_ligthing2 = modelowanie_log(n,train_ligthing2)
dobre_ligthing5 = modelowanie_log(n,train_ligthing5)
dobre_ligthing4 = modelowanie_log(n,train_ligthing4)
dobre_refrigerator =modelowanie_log(n,train_refrigerator)
dobre_microwave =modelowanie_log(n,train_microwave)

dobre = [dobre_ligthing2,dobre_ligthing5,dobre_ligthing4,dobre_refrigerator,dobre_microwave]
print("stworzone modele")
do_liczenia_mocy = [train_ligthing2[:4000],train_ligthing5[:4000],train_ligthing4[:4000],train_refrigerator[:4000],train_microwave[:4000]]
jaki_dobry = [0,1,2,3,4]

do_liczenia_mocy = []
jaki_dobry = []

train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}

dlugosc = 24
for _ in range(10):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 100
for _ in range(10):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 250
for _ in range(10):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 500
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 1000
for _ in range(3):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)


def co_wybierze(dobre,ktory_z_dobre,X):
    s0 = dobre[0][ktory_z_dobre[0]].score(X)
    s1 = dobre[1][ktory_z_dobre[1]].score(X)
    s2 = dobre[2][ktory_z_dobre[2]].score(X)
    s3 = dobre[3][ktory_z_dobre[3]].score(X)
    s4 = dobre[4][ktory_z_dobre[4]].score(X)

    d = {s0:0,s1:1,s2:2,s3:3,s4:4}
    return d[max([s0,s1,s2,s3,s4])]
for j in range(len(jaki_dobry)):
    for k in do_mocy.keys():
        if co_wybierze(dobre,k,do_liczenia_mocy[j])==jaki_dobry[j]:
            do_mocy[k]+=1
    print("pyk")

print(dict(sorted(do_mocy.items(), key=lambda item: item[1])))
print(len(jaki_dobry))






