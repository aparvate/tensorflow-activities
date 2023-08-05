import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, optimizers
#Initialize neetwork and training
EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.legacy.Adam()
VALIDATION_SPLIT=0.90
IMG_ROWS, IMG_COLS = 28, 28 #Input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10  #Number of outputs = number of digits

#Define the convolutional network 
def build(input_shape, classes):
    model = models.Sequential()

    #Add layers
    # CONV => RELU => POOL
    model.add(layers.Convolution2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Add more layers, increase amt
    # CONV => RELU => POOL
    model.add(layers.Convolution2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten => RELU layers
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    # a softmax classifier
    model.add(layers.Dense(classes, activation="softmax"))
    return model

#Shuffle, split, and reshape data
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

#Normalize, and change to float
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

#Init optimizer and model
model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
    metrics=["accuracy"])
model.summary()

#TensorBoard (just writes)
callbacks = [
  # Write TensorBoard logs to './logs' directory
  tf.keras.callbacks.TensorBoard(log_dir='./Chapter 3/activityOneLogs')
]

#Fit model to data
history = model.fit(X_train, y_train, 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])