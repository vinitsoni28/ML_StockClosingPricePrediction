#Importing the libraries
import math
import pandas as pd
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#Getting the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2013-1-1', end='2020-6-25')

#Showing the data
print(df)

#Visualizing using a graph
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=15)
plt.ylabel('Closing Price in USD', fontsize=15)
plt.show()

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Converting the dataframe to a numpy array
dataset = data.values

#Computing the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#Scaling the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#Creating the scaled training data set
train_data = scaled_data[0:training_data_len , : ]

#Splitting the data between x_train and y_train
x_train = []
y_train = []
for i in range (60, len(train_data)):
    x_train.append(train_data[i-60 : i, 0])
    y_train.append(train_data[i, 0])

#Converting x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building the LSTM Network Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Training the model
model.fit(x_train, y_train, batch_size=100, epochs=15)

#Creating a test dataset
test_data = scaled_data[training_data_len - 60: , : ]

#Creating the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len : , : ] #Getting the remaining rows
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60 : i, 0])

x_test = np.array(x_test)

#Reshaping the data in the shape accepted by LSTM (3-Dimensional)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Getting the models predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #Undo Scaling

#Calculating RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("RMSE: ", rmse)

#Creating the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualizing the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price in USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Trained', 'Val', 'Prediction'], loc='lower right')
plt.show()

#Showing the valid and predicted values
print(valid)

#Getting the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2013-1-1', end='2020-6-25')

#Creating a new dataframe
new_df = apple_quote.filter(['Close'])

#Getting the last 60-day closing price of the stock
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Creating an empty list
X_test = []

#Appending the data from past 60 days
X_test.append(last_60_days_scaled)

#Converting the X_test dataset to a numpy array
X_test = np.array(X_test)

#Reshaping the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Predicting scaled price
pred_price = model.predict(X_test)

#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print('25 June: ', pred_price)



