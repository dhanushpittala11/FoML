import pandas as pd
import numpy as np
from sklearn import *
from scipy.spatial import distance
import scipy.stats as st
# Loading the data and dropping the index axis
df = pd.read_csv('Datasets/kidney_disease.csv')
df  = df.drop(['id'],axis=1)


# separating data into different classes
real = ['sc','pot','hemo','rc', 'sg']
integer = ['age','bp','bgr','bu','sod','pcv','wc','su', 'al']
# binary = ['rbc', 'pc', 'pcc', 'pa', 'bgr', 'htn', 'dm','cad']
label = ['classification']
cat = list(set(df.columns) - set(real)-set(integer)-set(label))

# Removing parsing errors
df = df.replace('\t?',np.nan)
df = df.replace('\tyes','yes')
df = df.replace(' yes','yes')
df = df.replace('yes\t','yes')
df = df.replace('\tno','no')
df = df.replace('ckd\t','ckd')
df = df.replace('ckd',1)
df = df.replace('notckd',0)


# Filling the null values with mean you can also use other statistic like mode or median
for r in real:
    mean = np.array(df[r][~df[r].isna()]).astype('float').mean()
    df[r] = df[r].fillna(mean)
for i in integer:
    mean = np.array(df[i][~df[i].isna()]).astype('int').mean()
    df[i] = df[i].fillna(int(mean))
    
    
X_cat  = X[cat]
X_int = X[integer].astype('int64')
X_real = X[real]

X_cat = pd.get_dummies(X_cat, columns = X_cat.columns)
X_cat
#X_cat = X_cat.astype('bool')
X_cat

def find_minkowski(a, b):
    return distance.minkowski(a,b)
def find_canberra(a, b):
    return distance.canberra(a, b)
def find_rusrao(a, b):
    return distance.russellrao(a, b)
    
def find_distance(a_real, b_real, a_int, b_int, a_cat, b_cat):
    mink = find_minkowski(a_real, b_real)
    # print(mink)
    canb = find_canberra(a_int, b_int)
    #print(canb)
    rus = find_rusrao(a_cat, b_cat)
    #print(rus)
    return mink + canb + rus
    
# X_real = X_real.to_numpy()
# X_cat = X_cat.to_numpy()
# X_int = X_int.to_numpy()
# y = y
X_real = X_real.astype('float64')
X_int = X_int.astype('int64')
X_cat = X_cat.astype('bool')

X_train_real, X_test_real, X_train_int, X_test_int, X_train_cat, X_test_cat, y_train, y_test = model_selection.train_test_split(X_real, X_int, X_cat, y, random_state = 42, test_size = 0.33)

X_train_real

X_train_int

Xtrain_real,Xval_real, Xtrain_int,Xval_int,Xtrain_cat,Xval_cat,ytrain,yval = model_selection.train_test_split(X_train_real,X_train_int, X_train_cat, y_train, random_state = 42, test_size = 0.5)

print(Xtrain_real.shape)
print(X_test_real.shape)
print(Xval_real.shape)

display(X_test_real)
X_test_real.dtypes

Xtrain_cat = Xtrain_cat.to_numpy()
Xval_cat = Xval_cat.to_numpy()
Xtest_cat = X_test_cat.to_numpy()
Xtrain_int = Xtrain_int.to_numpy()
Xval_int = Xval_int.to_numpy()
Xtest_int = X_test_int.to_numpy()
Xtrain_real = Xtrain_real.to_numpy()
Xval_real  = Xval_real.to_numpy()
Xtest_real = X_test_real.to_numpy()
ytrain = ytrain.to_numpy()
yval = yval.to_numpy()
ytest = y_test.to_numpy()

print(type(distance.russellrao(Xtest_cat[0], Xtest_cat[0])))
print(distance.minkowski(Xtest_real[1], Xtest_real[2]))
distance.canberra(Xtest_int[1], Xtest_int[2])

def KNN(x_real, x_int, x_cat, X_train_real, X_train_int, X_train_cat,y_train, k=3):
    distances = []
    for i in range(X_train_real.shape[0]):
        distances.append(find_distance(x_real, X_train_real[i], x_int, X_train_int[i], x_cat, X_train_cat[i]))
    dist_array = np.array(distances)
    ind = np.argpartition(dist_array, k)
    return st.mode(y_train[ind[:k]]).mode[0][0]
    
label=  KNN(Xtest_real[1],Xtest_int[1],Xtest_cat[1], Xtrain_real, Xtrain_int, Xtrain_cat , ytrain)
print(label)
# find_distance(Xtest_real[0], Xtrain_real[0], Xtest_int[0], Xtrain_int[0], Xtest_cat[0], Xtrain_cat[0])

def find_accuracy(X_ds_real, X_ds_int, X_ds_cat, y_ds, k_nn):
    labels = []
    for i in range(X_ds_real.shape[0]):
        label = KNN(X_ds_real[i], X_ds_int[i], X_ds_cat[i], Xtrain_real, Xtrain_int, Xtrain_cat,ytrain, k=k_nn)
        labels.append(label)
    return metrics.explained_variance_score(y_ds, np.array(labels))
    
find_accuracy(Xtest_real, Xtest_int, Xtest_cat, ytest, 3)
find_accuracy(Xval_real, Xval_int, Xval_cat, yval, 3)

def validate(k_start, k_end, Xval_real, Xval_int, Xval_cat, yval):
    k_best = 1
    acc_best = 0
    for k in range(k_start, k_end,2):
        acc = find_accuracy(Xval_real, Xval_int, Xval_cat, yval, k_nn=k)
        print("Test Accuracy for k = ", k, "is = ", acc)
        if(acc >= acc_best):
            k_best = k
        else:
            continue
    return k_best
    
k_best = validate(1, 11, Xval_real, Xval_int, Xval_cat, yval)
print("the best k  = ", k_best)


X = df.drop(label,axis=1)
y = df[label]
