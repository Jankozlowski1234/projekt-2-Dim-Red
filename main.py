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
'''

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

'''



#druga czesc szukania najepszego modelu
'''

najlepsze_modele = [
(8,19,19,18,17),
(8,19,19,17,17),
(8,19,19,19,17),
(8,16,19,18,17),
(8,16,19,17,17),
(8,16,19,19,17),
(8,18,17,18,6),
(8,18,17,18,17),
(8,18,17,17,6),
(8,18,17,17,17),
(8,18,17,19,6),
(8,18,17,19,17),
(8,16,17,18,6),
(8,16,17,18,17),
(8,16,17,17,6),
(8,16,17,17,17),
(8,16,17,19,6),
(8,16,17,19,17),
(8,19,19,18,17),
(8,19,19,17,17)]




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



##plot power for different length of samples
'''
ile_do_testowania = 2001
trains = [train_ligthing2[ile_do_testowania:],train_ligthing5[ile_do_testowania:],train_ligthing4[ile_do_testowania:],
          train_refrigerator[ile_do_testowania:],train_microwave[ile_do_testowania:]]

The_best_components = (8, 18, 17, 19, 6)
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

print("poszlo")
##to many states
The_best_components = (19,19,19,19,19)
lst_the_best_model = []
for nr,tr in zip([0,1,2,3,4],trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)


dlugosci = np.array([5,10,18,24,30,50,75,100,125,150,175,200,225,250,275,300,350,400,500,750,1000,1250,1500,1750,1999])
df2 = pd.DataFrame( columns=['power', 'length',"model"])
for j in range(len(dlugosci)):
    df2.loc[j] = (sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j],lst_the_best_model,k = k),dlugosci[j],str(The_best_components))

print("poszlo")
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

print("poszlo")
##lights_small_rest_big
The_best_components = (2,2,2,19,19)
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

print("poszlo")
##another good one
The_best_components = (8, 16, 17, 18, 17)
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


print("poszlo")
##another good one
The_best_components = (8, 19, 19, 19, 17)
lst_the_best_model = []
for nr, tr in zip([0, 1, 2, 3, 4], trains):
    model = hmm.GaussianHMM(n_components=The_best_components[nr])  # czy jeszcze jakieś inne parametry?
    model.fit(tr)
    lst_the_best_model.append(model)

dlugosci = np.array(
    [5, 10, 18, 24, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000, 1250, 1500, 1750,
     1999])
df6 = pd.DataFrame(columns=['power', 'length', "model"])
for j in range(len(dlugosci)):
    df6.loc[j] = (
    sprawdz_moc_dla_ustalonej_dlugosci(dlugosci[j], lst_the_best_model, k=k), dlugosci[j], str(The_best_components))

df = pd.concat([df1,df2,df3,df4,df5,df6])

df.to_csv('dane_na_wykresy.csv', index=False,header=True)

'''

#second plots n vs
'''
trains = [train_ligthing2,train_ligthing5,train_ligthing4,train_refrigerator,train_microwave]
witch_device = {0:"ligthing2",1:"lighting5",2:"lighting4",3:"refrigerator",4:"microwave"}


ile_val = 0.9
df_2 = pd.DataFrame(columns=['n', "value",'kryt', "device"])
j = 0
for n in range(1,20):
    for dev in range(5):
        X_tr = trains[dev][:int(ile_val*len(trains[dev]))]
        X_val = trains[dev][int(ile_val*len(trains[dev])):]
        model = hmm.GaussianHMM(n_components=n)
        model.fit(X_tr)

        score = model.score(X_val)
        df_2.loc[j] = (n, score, "log_lig", witch_device[dev])
        j += 1

        score = model.aic(X_val)
        df_2.loc[j] = (n, score, "AIC", witch_device[dev])
        j += 1

        score = model.bic(X_val)
        df_2.loc[j] = (n, score, "BIC", witch_device[dev])
        j += 1

df_2.to_csv('dane_na_wykresy_2.csv', index=False,header=True)
'''

Ns = range(1,20)


ile_do_testowania = 1000
trains = [train_ligthing2[ile_do_testowania:],train_ligthing5[ile_do_testowania:],train_ligthing4[ile_do_testowania:],
          train_refrigerator[ile_do_testowania:],train_microwave[ile_do_testowania:]]


train_inaczej = {0:train_ligthing2,1:train_ligthing5,2:train_ligthing4,3:train_refrigerator,4:train_microwave}


do_liczenia_mocy = []
jaki_dobry = []
dlugosci = [24,50,75,100,125,150,175,200,300,500,750]
n_1 = 10
dlug_pow = [n_1 for _ in range(11)]
for dlugosc,f in zip(dlugosci,dlug_pow):
    if(dlugosc<ile_do_testowania):
        for _ in range(f):
            for i in range(5):
                k = random.randint(0,(ile_do_testowania-1-dlugosc))
                do_liczenia_mocy.append(train_inaczej[i][k:(k+dlugosc)])
                jaki_dobry.append(i)

d_naj_mod = {}
moce = {}
j =0
for n_lam,n_rest in itertools.product(Ns,Ns):
    moce[(n_lam,n_rest)] = 0
    mod = [n_lam,n_lam,n_lam,n_rest,n_rest]
    lst_mod = []
    for nr,tr in zip([0,1,2,3,4],trains):
        model = hmm.GaussianHMM(n_components=mod[nr])
        model.fit(tr)
        lst_mod.append(model)
    d_naj_mod[(n_lam,n_rest)] = lst_mod[:]
    j += 1
    if j%25 ==0:
        print(f"stworzono modele {j}/{400}")




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


df_3 = pd.DataFrame(columns=['n_lamp',"n_rest",'power'])
j = 0
for k in moce.keys():
    df_3.loc[j] = (k[0],k[1],moce[k]/len(jaki_dobry))
    j+=1

df_3.to_csv('dane_na_wykresy_3.csv', index=False,header=True)

















