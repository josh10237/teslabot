# Import libraries
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import datetime

dataset = pd.read_csv(r'C:\Users\riley\Desktop\teslabot\Google_Stock_Price_Train.csv', index_col="Date", parse_dates=True)

head = dataset.head()
# print(head)

dataset.isna().any()

# dataset['Open'].plot(figsize=(16,6))
# pyplot.show()

# convert column "a" of a DataFrame
dataset['Volume'] = dataset['Volume'].astype(float)

# dataset.info()

# 7 day rolling mean
head_roll_mean = dataset.rolling(7).mean().head(20)
# print(head_roll_mean)

# dataset['Open'].plot(figsize=(16,6))
# dataset.rolling(window=30).mean()['Close'].plot()
# pyplot.show()

dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close', 'Close: 30 Day Mean']].plot(figsize=(16, 6))
# pyplot.show()

training_set = dataset['Open']
training_set = pd.DataFrame(training_set)

# Data cleaning
dataset.isna().any() # watch out here

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
x_train, y_train = [], []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Import Keras library/packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Intializing RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv(r'C:\Users\riley\Desktop\teslabot\Google_Stock_Price_Test.csv', index_col='Date', parse_dates=True) #add in test data

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_test.head()
dataset_test.info()

dataset_test['x'] = dataset_test['x'].astype(float) # fill in data changing type for x

test_set = dataset_test['Open']
test_set = pd.DataFrame(test_set)

test_set.info()

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset['Open'], dataset_test['Open'], axis=0))
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range (60, 80):
    x_test = x_test.appen(inputs[i-60:i, 0])
x_test = np.array(x_test)
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_stock_price = pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Tesla Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Tesla Stock Price')
plt.tile('Tesla Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()