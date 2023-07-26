import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras as K
from keras.layers import Dense

#Generate random data
np.random.seed(np.random.randint(0,500000))
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20,50, size = len(area))
data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns=['area','price'])
plt.scatter(data['area'], data['price'])
plt.show()

#Normalize
data = (data - data.min()) / (data.max() - data.min())

#Build model
model = K.Sequential([Dense(1, input_shape = [1,], name = 'first-layer', activation=None)])
model.summary()

#Compile model with Gradient Descent optimizer
model.compile(loss='mean_squared_error', optimizer='sgd')

#Fit data to model
model.fit(x=data['area'],y=data['price'], epochs=200, batch_size=32, verbose=1, validation_split=0.2)

#New price prediction
y_pred = model.predict(data['area'])

#plot predicted prices
plt.plot(data['area'], y_pred, color='red',label="Predicted Price")
plt.scatter(data['area'], data['price'], label="Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show() 