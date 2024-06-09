import argparse
import pandas as pd
import numpy as np
import math
from hmmlearn import hmm

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

train_ligthing2 = train[:,0]
train_ligthing5 = train[:,0]
train_ligthing4 = train[:,0]
train_refrigerator = train[:,0]
train_microwave = train[:,0]


print(train_ligthing4)