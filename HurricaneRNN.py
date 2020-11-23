#This is the Hurricane Economic Damage Predicition Code

#import the needed packages
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler # for normalizing the data
from sklearn.metrics import mean_squared_error

#loading the dataset from location and replacing errors with 0's
names=['Year','Land_Month','Land_Day','Land_Year','End_Month','End_Day','End_Year','Vmax (knot)','Vmax (mph)','HII	Category',
       'Landfall_Lat','Landfall_Long','Landfall_NDVI','Landfall_NDWI', 'StartLandVal'
       ,'P2_Lat','P2_Long',	'P2_NDVI','P2_NDWI','P2_Val','P3_Lat','P3_Long','P3_NDVI','P3_NDWI','P3_Val',
       'P4_Lat]','P4_Long','P4_NDVI','P4_NDWI',	'P4_Val','P5_Lat',	'P5_Long',	'P5_NDVI',	'P5_NDWI',
       'P5_Val','P6_Lat','P6_Long',	'P6_NDVI','P6_NDWI','P6_Val','P7_Lat','P7_Long','P7_NDVI','P7_NDWI',
       'P7_Val', 'P8_Lat',	'P8_Long',	'P8_NDVI', 'P8_NDWI','P8_Val','End_Lat','End_Long',	'End_NDVI',
       'End_NDWI','End_Val','Tot_Val']

#taken from the database schema
desktop_location =  #the location of the file with the actual data (data should be a CSV file)
dataset=pd.read_csv(desktop_location,names=names)
dataset.fillna(0, inplace=True)
dataset.replace(numpy.nan,0)
print(f'The shape of the data set is: {dataset.shape}')

#creating the data and the labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 55].values
features = len(X[0])
labels = numpy.unique(y)

#Splitting the data into testing and training groups and labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Normalizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(f'The training data length is: {len(X_train)}, the testing data length is: {len(X_test)}')
print(f'The shape of the training data is: {X_train.shape}')
print(f'The shape of the test data set is: {X_test.shape}')

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i: (i + time_step), :]
        a=numpy.array(a)
        dataX.append(a)
        dataY.append(dataset[i + time_step, -1])
        #print(f'the length of dataX is: {len(dataX)}')
    # data format [samples, timestep]
    dataX = numpy.array(dataX)
    # data format [samples, timestep, feature]
    print(f'datax prereshape-shape: {dataX.shape}')
    dataX = dataX.reshape(dataX.shape[0], time_step, features)
    print('dataX shape: ', dataX.shape)
    dataY = numpy.array(dataY)
    print('dataY shape: ', dataY.shape)
    #exit()
    return dataX, dataY

time_step = 3
trainX, trainY = create_dataset(X_train, time_step)
testX, testY = create_dataset(X_test, time_step)

#Creating the RNN and LSTM Neural Networks
RNN = Sequential()
RNN.add(layers.SimpleRNN(100, input_shape=(time_step, features)))
RNN.add(layers.Dense(1))

LSTM = Sequential()
LSTM.add(layers.LSTM(512, input_shape=(time_step, features)))
LSTM.add(layers.Dense(1))

#Using the mean squared error to determine our training value loss
RNN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


#establishing the training history for both the RNN and LSTM
history_RNN = RNN.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=1)
history_LSTM = LSTM.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=1)

#Plotting our training history for both the LSTM and RNN
plt.plot(history_RNN.history['val_loss'], label='val_loss')
plt.plot(history_RNN.history['loss'], label='loss')
plt.title('RNN Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(history_LSTM.history['val_loss'], label='val_loss')
plt.plot(history_LSTM.history['loss'], label='loss')
plt.title('LSTM Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

# make predictions
trainPredict = RNN.predict(trainX) #18-1 shape
trainPredict = [[ x[0] for i in range(0, 55) ] for x in trainPredict]
trainPredict = numpy.array(trainPredict)
print('the train predict shape is:', trainPredict.shape)

testPredict = RNN.predict(testX)
testPredict = [[ x[0] for i in range(0, 55) ] for x in testPredict]
testPredict = numpy.array(testPredict)
print('the test predict shape is:', testPredict.shape)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)

trainY = [[ x for i in range(0, 55) ] for x in trainY]
orginal_trainY = scaler.inverse_transform(trainY)
print('the original train Y shape is: ', orginal_trainY.shape)

testPredict = scaler.inverse_transform(testPredict)

testY = [[ x for i in range(0, 55) ] for x in testY]
orginal_testY = scaler.inverse_transform(testY)
print('the original test Y shape is: ', orginal_testY.shape)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(orginal_trainY[:,0], trainPredict[:,0]))
print('Train Score MSR: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(orginal_testY[:,0], testPredict[:,0]))
print('Test Score MSR: %.2f RMSE' % (testScore))
