#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:41:54 2018

@author: sharad
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np



import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2000,1,1)
end = dt.datetime(2017,12,31)

#df = web.DataReader('AAPL','google', start, end)
#print(df.head(10))
#print(df.tail(10))

#df.to_csv('APPLE.csv')

df = pd.read_csv('APPLE.csv', parse_dates= True, index_col = 0)

#df.plot()
#df['Close'].plot()

#plt.show()


'another code for importing many comanys stocks'
#tickers = ["AAPL","GOOG","MSFT","XOM","BRK-A","FB","JNJ","GE","AMZN","WFC"]
#import pandas_datareader.data as web
#import datetime as dt
#end = dt.datetime.now().strftime("%Y-%m-%d")
#start = (dt.datetime.now()-dt.timedelta(days=365*3)).strftime("%Y-%m-%d")
#%matplotlib inline
#import matplotlib.pyplot as plt
#import pandas as pd
#data = []
#for ticker in tickers:
#    sub_df = web.get_data_yahoo(ticker, start, end)
#    sub_df["name"] = ticker
 #   data.append(sub_df)
#data = pd.concat(data)
 
 
 
 
 df = df.values
 df = df[:,0:4]
 
 
 n= df.shape[0]
 p= df.shape[1]
 
 
 
train_start= 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = df[np.arange(train_start,train_end),:]
data_test = df[np.arange(test_start,test_end),:]



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


X_train= data_train[:,0:3]
y_train = data_train[:,3]
X_test = data_test[:, 0:3]
y_test = data_test[:, 3]

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()
 



# Import TensorFlow
import tensorflow as tf




# Model architecture parameters
n_stocks = X_train.shape[1]
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1


X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))


hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

#cost function
mse = tf.reduce_mean(tf.squared_difference(out,Y))

#optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

#initializers

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode= "fan_avg",distribution = "uniform",scale= sigma)
bias_initializer = tf.zeros_initializer()

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()


batch_size = 50
mse_train=[]
mse_test=[]


# Number of epochs and batch size
epochs = 100000


for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            mse_train.append(net.run(mse, feed_dict = {X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict = {X: X_test, Y: y_test}))
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
            plt.show()
            graph = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(graph)
            
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)

pred = np.reshape(pred,(800,1))







