import argparse
import pandas as pd
import numpy as np

from hmmlearn import hmm
import random
import itertools

import os



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




train = pd.read_csv(TRAIN_NAME).iloc[:,1:]

train_ligthing2 = train.iloc[:,0].values.reshape(-1, 1)
train_ligthing5 = train.iloc[:,1].values.reshape(-1, 1)
train_ligthing4 = train.iloc[:,2].values.reshape(-1, 1)
train_refrigerator = train.iloc[:,3].values.reshape(-1, 1)
train_microwave = train.iloc[:,4].values.reshape(-1, 1)


train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}


trains = [train_ligthing2,train_ligthing5,train_ligthing4,train_refrigerator,train_microwave]

The_best_components = (8,8,9,10,7)
lst_the_best_model = []
for nr,tr in zip([0,1,2,3,4],trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)

def jake_urzadzenie(lst_models,X):
    s0 = lst_models[0].score(X)
    s1 = lst_models[1].score(X)
    s2 = lst_models[2].score(X)
    s3 = lst_models[3].score(X)
    s4 = lst_models[4].score(X)
    d = {s0: 0, s1: 1, s2: 2, s3: 3, s4: 4}
    return d[max([s0, s1, s2, s3, s4])]

witch_device = {0:"ligthing2",1:"lighting5",2:"lighting4",3:"refrigerator",4:"microwave"}

#testing_data


with open(RESULT_NAME, "w") as file1:
    # Writing data to a file
    file1.write("file , dev_classified \n")
    for filename in os.listdir(TEST_NAME):
        df = pd.read_csv(os.path.join(os.getcwd(),TEST_NAME ,filename))
        df = df.iloc[:, 1].values.reshape(-1, 1)
        jaki = jake_urzadzenie(lst_the_best_model,df)
        nazwa = witch_device[jaki]
        file1.write(f"{filename},{nazwa} \n")




