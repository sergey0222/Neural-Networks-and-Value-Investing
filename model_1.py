# this module creates input data and feeds it to Regression model of fully connected neural network
# the input data is tag NetIncomeLoss as it appears in 2011-2017 Full Year Reports
# labled data is the same tag for 2018 Full Year Report 
# the task is to predict Net Income based on previous year values

import datetime
import numpy as np
import csv
import datasets_lib as ds
import math

#import tensorflow as tf
#from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1

# store starting time for calculating total processing time
time_start = datetime.datetime.now()

# set path to folder Datasets/
path = 'C:/Users/belyisn.N0COSA/Desktop/Files/DataSets/'

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
adsh_cik_year = np.zeros((len(index_adsh),3), dtype=int)

# for each adsh get cik, sic and Full Report Year
# as sic should be reindexed initialize needed objects
sic_dic = {}
next_available_index = 0

with open(path + 'filter_1/filter_1_sub.txt') as f:
        f_object = csv.reader(f, delimiter='\t')
        for row in f_object:
            adsh_int = int(row[0])
            cik_int = int(row[1])
            
            sic_str = row[3]
            index, next_available_index = ds.index_by_tree (sic_str, sic_dic, next_available_index)
            
            fy_int = int(row[27])
            
            adsh_cik_year[adsh_int,0] = cik_int
            adsh_cik_year[adsh_int,1] = fy_int
            adsh_cik_year[adsh_int,2] = index

# create array for keeping sic one hot vectors
XY_onehot = np.zeros((len(index_cik), next_available_index), dtype = float)            

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
               
               index = adsh_cik_year[adsh_int,2]
               XY_onehot[r,index] = 1

# define rows where values exist for all years in a range
mask1D = np.all(XY_bool, axis=1, keepdims = True)

# create list of eligible ciks
elig_cik_list = np.nonzero(mask1D)[0]

# extact only eligible rows
XY = XY_float[elig_cik_list,:]
XY_onehot = XY_onehot[elig_cik_list,:]

# shuffle the array
#np.random.shuffle(XY)

# define a year to predict
year_to_predict = 2018

# cut only needed years
XY_cut = XY[:,0:(year_to_predict - first + 1)]

X = XY_cut[:,0:-1]
Y = XY_cut[:,-1:]

# normalize X (using X data only!)
mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - mean
maximum = np.max(np.abs(X_centered), axis=1, keepdims=True)
X_normilized = X_centered / maximum + 10
X_onehot_normilized = np.concatenate((XY_onehot,X_normilized),axis=1)

# normilize Y using X data (not Y!)
Y_normilized = (Y - mean) / maximum + 10

# splet between train and dev sets
# calculate number of traning examples
m = np.shape(X_normilized)[0]   
# set percentage for test set
test_percent = 10
m_test = math.floor(test_percent/100*m)
m_train = m - m_test

# Input and labels for neural network
X_train = X_onehot_normilized[0:m_train,:]
Y_train = Y_normilized[0:m_train,:]

X_test = X_onehot_normilized[m_train:,:]
Y_test = Y_normilized[m_train:,:]
 
# build a model
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001), input_shape=(np.shape(X_train)[1],)))
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.001)))
model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.001)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='RMSProp')
model.fit(X_train, Y_train, epochs=1000, batch_size=32)

# evaluate result
loss = model.evaluate(X_test, Y_test)
print('Test set loss is:', loss)

#body = np.mean(np.abs(Y_test))
#print('Error on a test set is', round(loss/body*100,1), '%')

'''
# let us examine the results manually
# define which test example to choose
test_example = 13
print('Test example input is:', X_test[test_example])
print('Test example output is', Y_test[test_example])

prediction = model.predict(X_test)
print('Prediction is:', prediction[test_example,0])   
'''

# processing time
print('time elapsed - ', datetime.datetime.now() - time_start)    
            
           

            
            
            
            
            
