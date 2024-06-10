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
    return best_three_values




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
    best_three_values = values[0:3]
    return best_three_values

def modelowanie_BIC(n, X):
    X = X[4001:]
    X_train = X[:int(0.9 * len(X))]
    X_validate = X[int(0.9 * len(X)):]
    models = {}
    for idx in range(1, n):
        model = hmm.GaussianHMM(n_components=idx)  # czy jeszcze jakieś inne parametry?
        model.fit(X_train)
        Bic = model.bic(X_validate)
        models[Bic] = model
    models = dict(sorted(models.items()))
    values = list(models.values())
    best_three_values = values[0:3]
    return best_three_values

#modelowanie_log(11, train_microwave)
#### cuting the data


#k - dlugosc danych, N - ile powtorzen col - ktora kolumna(urzadzenie)
def stworz_nowe(train,k = 1000,col = 0,N = 100):
    cutted = train.iloc[:,col].values
    n = len(cutted)
    od_kad = np.random.randint(low = 0, high=n-k-1, size=N)

    return None


#liczenie jaki modele maja najwieksza moc


#### finding the best one
a=[0,1,2]

listy = list(itertools.product(a,a,a,a,a))
do_mocy = {}
for lis in listy:
    do_mocy[tuple(lis)] = 0

n = 20
dobre_ligthing2 = modelowanie_log(n,train_ligthing2)
dobre_ligthing5 = modelowanie_log(n,train_ligthing5)
dobre_ligthing4 = modelowanie_log(n,train_ligthing4)
dobre_refrigerator =modelowanie_log(n,train_refrigerator)
dobre_microwave =modelowanie_log(n,train_microwave)

dobre = [dobre_ligthing2[:],dobre_ligthing5[:],dobre_ligthing4[:],dobre_refrigerator[:],dobre_microwave[:]]



dobre_ligthing2 = modelowanie_AIC(n,train_ligthing2)
dobre_ligthing5 = modelowanie_AIC(n,train_ligthing5)
dobre_ligthing4 = modelowanie_AIC(n,train_ligthing4)
dobre_refrigerator =modelowanie_AIC(n,train_refrigerator)
dobre_microwave =modelowanie_AIC(n,train_microwave)

dobre_AIC = [dobre_ligthing2[:],dobre_ligthing5[:],dobre_ligthing4[:],dobre_refrigerator[:],dobre_microwave[:]]

dobre_ligthing2 = modelowanie_BIC(n,train_ligthing2)
dobre_ligthing5 = modelowanie_BIC(n,train_ligthing5)
dobre_ligthing4 = modelowanie_BIC(n,train_ligthing4)
dobre_refrigerator =modelowanie_BIC(n,train_refrigerator)
dobre_microwave =modelowanie_BIC(n,train_microwave)

dobre_BIC = [dobre_ligthing2[:],dobre_ligthing5[:],dobre_ligthing4[:],dobre_refrigerator[:],dobre_microwave[:]]
print("stworzone modele")


do_liczenia_mocy = []
jaki_dobry = []

train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}

dlugosc = 24
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 50
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 100
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 150
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)

dlugosc = 250
for _ in range(5):
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
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 2000
for _ in range(5):
    for i in range(5):
        k = random.randint(0,(4000-1-dlugosc))
        do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
        jaki_dobry.append(i)
dlugosc = 3000
for _ in range(5):
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

i = 0
for j in range(len(jaki_dobry)):
    for k in do_mocy.keys():
        if co_wybierze(dobre,k,do_liczenia_mocy[j])==jaki_dobry[j]:
            do_mocy[k]+=1
    i+=1
    if i % 10 ==0:
        print(f"pyk {i}/{len(jaki_dobry)}")
sorted_dict = dict(sorted(do_mocy.items(), key=lambda item: item[1],reverse= True))


m= max(sorted_dict.values())
for w,x in sorted_dict.items():
    if x==m:
        print(f"log accuracy = {m}/{len(jaki_dobry)}")
        for dev in range(5):
            print(f"device {dev}, numer of components: {dobre[dev][w[dev]].n_components}")
        print("--------")




for k in do_mocy.keys():
    do_mocy[k] = 0

i = 0
for j in range(len(jaki_dobry)):
    for k in do_mocy.keys():
        if co_wybierze(dobre_AIC,k,do_liczenia_mocy[j])==jaki_dobry[j]:
            do_mocy[k]+=1
    i+=1
    if i % 10 ==0:
        print(f"pyk {i}/{len(jaki_dobry)}")
sorted_dict = dict(sorted(do_mocy.items(), key=lambda item: item[1],reverse= True))


m= max(sorted_dict.values())
for w,x in sorted_dict.items():
    if x==m:
        print(f"AIC accuracy = {m}/{len(jaki_dobry)}")
        for dev in range(5):
            print(f"device {dev}, numer of components: {dobre[dev][w[dev]].n_components}")
        print("--------")




for k in do_mocy.keys():
    do_mocy[k] = 0

i = 0
for j in range(len(jaki_dobry)):
    for k in do_mocy.keys():
        if co_wybierze(dobre_BIC, k, do_liczenia_mocy[j]) == jaki_dobry[j]:
            do_mocy[k] += 1
    i += 1
    if i % 10 == 0:
        print(f"pyk {i}/{len(jaki_dobry)}")
sorted_dict = dict(sorted(do_mocy.items(), key=lambda item: item[1], reverse=True))

m = max(sorted_dict.values())
for w, x in sorted_dict.items():
    if x == m:
        print(f"BIC accuracy = {m}/{len(jaki_dobry)}")
        for dev in range(5):
            print(f"device {dev}, numer of components: {dobre[dev][w[dev]].n_components}")
        print("--------")





#druga czesc szukania najepszego modelu
'''

najlepsze_modele = [(8,6,9,10,7),(8,6,9,10,8),(8,8,9,9,7),(8,8,9,9,8),(8,8,9,10,7),
                    (8,8,9,10,8),(4,6,8,6,6),(4,8,8,6,6),(6,8,10,9,7),(6,8,10,10,7)]


ile_do_testowania = 751
trains = [train_ligthing2[ile_do_testowania:],train_ligthing5[ile_do_testowania:],train_ligthing4[ile_do_testowania:],
          train_refrigerator[ile_do_testowania:],train_microwave[ile_do_testowania:]]


d_naj_mod = {}
moce = {}
for mod in najlepsze_modele:
    moce[mod] = 0
    lst_mod = []
    for nr,tr in zip([0,1,2,3,4],trains):
        model = hmm.GaussianHMM(n_components=mod[nr])  # czy jeszcze jakieś inne parametry?
        model.fit(tr)
        lst_mod.append(model)
    d_naj_mod[mod] = lst_mod[:]
    moce[mod] = 0
print("stworzono modele")


do_liczenia_mocy = []
jaki_dobry = []

train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}

dlugosci = [24,50,75,100,125,150,175,200,300,500,750,999]
dlug_pow = [10,10,10,10,10,10,10,10,10,10,5,2]
for dlugosc,f in zip(dlugosci,dlug_pow):
    if(dlugosc<ile_do_testowania):
        for _ in range(f):
            for i in range(5):
                k = random.randint(0,(ile_do_testowania-1-dlugosc))
                do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
                jaki_dobry.append(i)



def jake_urzadzenie(lst_models,X):
    s0 = lst_models[0].score(X)
    s1 = lst_models[1].score(X)
    s2 = lst_models[2].score(X)
    s3 = lst_models[3].score(X)
    s4 = lst_models[4].score(X)
    d = {s0: 0, s1: 1, s2: 2, s3: 3, s4: 4}
    return d[max([s0, s1, s2, s3, s4])]
i = 0
for j in range(len(jaki_dobry)):
    for k in moce.keys():
        if jake_urzadzenie(d_naj_mod[k],do_liczenia_mocy[j]) == jaki_dobry[j]:
            moce[k] += 1
    i += 1
    if i % 25 == 0:
        print(f"pyk {i}/{len(jaki_dobry)}")
sorted_dict = dict(sorted(moce.items(), key=lambda item: item[1], reverse=True))



m = max(sorted_dict.values())
for w, x in sorted_dict.items():
    if x == m:
        print(f"accuracy = {m}/{len(jaki_dobry)}")
        print(f"nr of components:{w}")
        print("--------")

'''



'''


ile_do_testowania = 2001
trains = [train_ligthing2[ile_do_testowania:],train_ligthing5[ile_do_testowania:],train_ligthing4[ile_do_testowania:],
          train_refrigerator[ile_do_testowania:],train_microwave[ile_do_testowania:]]

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


train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}

def sprawdz_moc_dla_ustalonej_dlugosci(dlugosc,lst_the_best_model,k = 10):
    do_liczenia_mocy = []
    jaki_dobry = []

    for _ in range(k):
        for i in range(5):
            k = random.randint(0,(ile_do_testowania-1-dlugosc))
            do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
            jaki_dobry.append(i)

    s = 0
    for j in range(len(jaki_dobry)):
        if jake_urzadzenie(lst_the_best_model, do_liczenia_mocy[j]) == jaki_dobry[j]:
            s += 1

    return s/len(jaki_dobry)
k = 100
dlugosci = np.array([5,10,18,24,30,50,75,100,125,150,175,200,225,250,275,300,350,400,500,750,1000,1250,1500,1750,1999])
df1 = pd.DataFrame( columns=['power', 'length',"model"])
for j in range(len(dlugosci)):
    df1.loc[j] = (sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j],lst_the_best_model,k = k),dlugosci[j],str(The_best_components))


##to many states
The_best_components = (10,10,10,10,10)
lst_the_best_model = []
for nr,tr in zip([0,1,2,3,4],trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)


dlugosci = np.array([5,10,18,24,30,50,75,100,125,150,175,200,225,250,275,300,350,400,500,750,1000,1250,1500,1750,1999])
df2 = pd.DataFrame( columns=['power', 'length',"model"])
for j in range(len(dlugosci)):
    df2.loc[j] = (sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j],lst_the_best_model,k = k),dlugosci[j],str(The_best_components))

##to less
The_best_components = (2,2,2,2,2)
lst_the_best_model = []
for nr, tr in zip([0, 1, 2, 3, 4], trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)

dlugosci = np.array(
    [5, 10, 18, 24, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000, 1250, 1500, 1750,
     1999])
df3 = pd.DataFrame(columns=['power', 'length', "model"])
for j in range(len(dlugosci)):
    df3.loc[j] = (
    sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j], lst_the_best_model, k=k), dlugosci[j], str(The_best_components))

##lights_small_rest_big
The_best_components = (2,2,2,10,10)
lst_the_best_model = []
for nr, tr in zip([0, 1, 2, 3, 4], trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)

dlugosci = np.array(
    [5, 10, 18, 24, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000, 1250, 1500, 1750,
     1999])
df4 = pd.DataFrame(columns=['power', 'length', "model"])
for j in range(len(dlugosci)):
    df4.loc[j] = (
    sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j], lst_the_best_model, k=k), dlugosci[j], str(The_best_components))


##another good one
The_best_components = (6, 8, 10, 10, 7)
lst_the_best_model = []
for nr, tr in zip([0, 1, 2, 3, 4], trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)

dlugosci = np.array(
    [5, 10, 18, 24, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000, 1250, 1500, 1750,
     1999])
df5 = pd.DataFrame(columns=['power', 'length', "model"])
for j in range(len(dlugosci)):
    df5.loc[j] = (
    sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j], lst_the_best_model, k=k), dlugosci[j], str(The_best_components))



df = pd.concat([df1,df2,df3,df4,df5])

df.to_csv('dane_na_wykresy.csv', index=False,header=True)
'''