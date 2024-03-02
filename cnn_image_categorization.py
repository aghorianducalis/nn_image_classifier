# convolutional neural network model for image classification
# https://www.cs.toronto.edu/~kriz/cifar.html

from __future__ import print_function

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import utils
# pip install pydot graphviz
# sudo apt install graphviz
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

# Training parameters
batch_size = 128
nb_classes = 10
nb_epoch = 10

img_rows, img_cols = 32, 32
img_channels = 3

print('Loading data...')

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)

X_train = (X_train.astype('float32') - 28) / 128
X_test = (X_test.astype('float32') - 28) / 128

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('Build model...')

# Define the sequential model
model = Sequential()

# Add Convolutional layers
model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_rows, img_cols, img_channels)))
model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
# Add MaxPooling2D layer to reduce spatial dimensions
model.add(MaxPooling2D(pool_size=(2, 2)))
# Apply dropout regularization to prevent overfitting
model.add(Dropout(0.25))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the feature maps
model.add(Flatten())
# Add fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# Output layer with softmax activation for classification
model.add(Dense(nb_classes, activation='softmax'))

sgd = SGD(
    learning_rate=0.01,
    weight_decay=1e-6,
    momentum=0.9,
    nesterov=True,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

plot_model(model, to_file='visualization/model_image_categorization.png')

print('Train...')

# Fit the model to the training data
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_test, Y_test),
    shuffle=True,
    callbacks=[ModelCheckpoint('weights/cifar.h5', save_best_only=True)]
)
