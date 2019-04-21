# this module creates input data and feeds it to Regression model of fully connected neural network
# the input data is tag NetIncomeLoss as it appears in 2011-2017 Full Year Reports
# labled data is the same tag for 2018 Full Year Report 
# the task is to predict Net Income based on previous year values

import datetime
import numpy as np
import csv
import datasets_lib as ds
import math
import pdb
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1

# store starting time for calculating total processing time
time_start = datetime.datetime.now()

# set path to folder Datasets/
path = 'D:/DataSets/'

# the tag to be used for input data
tag_org = 'NetIncomeLoss'

# define years taking part in X and Y creation 
first = int(2011)
last = int(2018)

# load important lists
index_cik=ds.list_from_file(path+'/reindexed/index_cik.txt')
index_adsh=ds.list_from_file(path+'/reindexed/index_adsh.txt')
index_tag=ds.list_from_file(path+'/reindexed/index_tag.txt')

# get reindexed version of the tag
tag_reindexed_int = index_tag.index(tag_org)

# create supporting arrays
XY_bool = np.zeros((len(index_cik), last-first+1), dtype = bool)
XY_float = np.zeros((len(index_cik), last-first+1), dtype = float)

# create empty array which will hold cik and FY year for each adsh
adsh_cik_year = np.zeros((len(index_adsh),2), dtype=int)

# for each adsh get cik and Full Report Year

with open(path + 'filter_1/filter_1_sub.txt') as f:
        f_object = csv.reader(f, delimiter='\t')
        for row in f_object:
            adsh_int = int(row[0])
            cik_int = int(row[1])          
            fy_int = int(row[27])
            
            adsh_cik_year[adsh_int,0] = cik_int
            adsh_cik_year[adsh_int,1] = fy_int

# look through filter_1_num.txt for all eppearances of the tag
with open(path + 'filter_1/filter_1_num.txt') as f:
        f_object = csv.reader(f, delimiter='\t')
        for row in f_object:
           tag = row[1]
           if int(tag) == tag_reindexed_int:
               adsh_int = int(row[0])
               cik_int = adsh_cik_year[adsh_int,0]
               fy_int = adsh_cik_year[adsh_int,1]
               value_float = float(row[7])
               
               # define row and column numbers
               r = cik_int
               c = fy_int - first
               
               # check if this field is being overwitten
               if XY_bool[r,c] == True:
                   print('Warning! Field is overwritten!')
               
               # mark this field as written
               XY_bool[r,c] = True
               
               # write down the correpsonding value
               XY_float[r,c] = value_float

# define a year to predict
year_to_predict = 2018

# cut only needed years
XY_bool_cut = XY_bool[:,0:(year_to_predict - first + 1)]
XY_float_cut = XY_float[:,0:(year_to_predict - first + 1)]

# define rows where values exist for all years in a range
mask1D = np.all(XY_bool_cut, axis=1, keepdims = True)

# create list of eligible ciks
elig_cik_list = np.nonzero(mask1D)[0]

# extact only eligible rows
XY = XY_float_cut[elig_cik_list,:]

# shuffle the array
np.random.shuffle(XY)

X = XY[:,0:-1]
Y = XY[:,-1:]

# scale X and Y (using X data only!)
maximum = np.max(np.abs(X), axis=1, keepdims=True)
X = X / maximum
Y = Y / maximum

# here the training stars with X and Y as input (pass1)
# set percentage for test set and calculate number of traning examples
test_percent = 20
m = np.shape(X)[0]   
m_test = math.floor(test_percent/100*m)
m_train = m - m_test

# Input and labels for neural network
X_train = X[0:m_train,:]
Y_train = Y[0:m_train,:]
X_test = X[m_train:,:]
Y_test = Y[m_train:,:]
 
# model
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001), input_shape=(np.shape(X)[1],)))
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001)))
model.add(Dense(1, kernel_regularizer=l1(0.001)))
model.compile(loss='mean_absolute_error', optimizer='RMSProp')
model.fit(X_train, Y_train, epochs=500, batch_size=32)

# evaluate result
loss_train = model.evaluate(X_train, Y_train)
loss_test = model.evaluate(X_test, Y_test)

# count difference between labes and predicted values
Ypred = model.predict(X)
delta = np.abs(Y - Ypred)
delta_sorted = np.sort(delta,axis=0)

# find training exmples eligible for pass2
mask_pass2 = delta <=  0.5
elig_pass2 = np.nonzero(mask_pass2)[0]

X = X[elig_pass2,:]
Y = Y[elig_pass2,:]

# pass2
# set percentage for test set and calculate number of traning examples
test_percent = 20
m = np.shape(X)[0]   
m_test = math.floor(test_percent/100*m)
m_train = m - m_test

# Input and labels for neural network
X_train = X[0:m_train,:]
Y_train = Y[0:m_train,:]
X_test = X[m_train:,:]
Y_test = Y[m_train:,:]
 
# model
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001), input_shape=(np.shape(X)[1],)))
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001)))
model.add(Dense(1, kernel_regularizer=l1(0.001)))
model.compile(loss='mean_absolute_error', optimizer='RMSProp')
model.fit(X_train, Y_train, epochs=500, batch_size=32)

# evaluate result
loss_train2 = model.evaluate(X_train, Y_train)
loss_test2 = model.evaluate(X_test, Y_test)

# count difference between labes and predicted values
Ypred = model.predict(X)
delta2 = np.abs(Y - Ypred)
delta_sorted2 = np.sort(delta2,axis=0)

plt.ylim(0,1)
plt.plot(delta_sorted)  
plt.plot(delta_sorted2)  
print('training loss, test loss for pass1:', round(loss_train,2), round(loss_test,2))
print('training loss, test loss for pass2:', round(loss_train2,2), round(loss_test2,2))

# processing time
print('time elapsed - ', datetime.datetime.now() - time_start)
