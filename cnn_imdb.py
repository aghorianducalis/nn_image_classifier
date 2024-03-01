# Solving the applied problem of determining the tonality of a comment using recurrent neural networks

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint

# Maximum number of words to consider as features
max_features = 20000
# Maximum length of sequences
max_length = 100
# Dimension of the embedding space
embedding_size = 128

# Parameters for the Convolutional Neural Network (CNN)
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM parameters
lstm_output_size = 70

# Training parameters
batch_size = 30
nb_epoch = 2

print('Loading data...')
# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print(len(X_test), 'Pad sequences (samples x time)')
# Pad sequences to ensure uniform length
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

# Define the sequential model
model = Sequential()
# Add an Embedding layer to convert words into dense vectors of fixed size
model.add(Embedding(max_features, embedding_size, input_length=max_length))
# Apply dropout regularization to prevent overfitting
model.add(Dropout(0.25))
# Add a 1D Convolutional layer for feature extraction
model.add(Conv1D(filters=nb_filter,
                 kernel_size=filter_length,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 ))
# Apply MaxPooling to down-sample the feature maps
model.add(MaxPooling1D(pool_size=pool_length))
# Add a Long Short-Term Memory (LSTM) layer for sequence modeling
model.add(LSTM(lstm_output_size))
# Add a Dense layer for classification (output layer)
model.add(Dense(1))
# Apply sigmoid activation function to produce probabilities
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

# checkpoint callback will save the best model based on the validation accuracy automatically during training
checkpoint = ModelCheckpoint('weights/bi_lstm.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Fit the model to the training data
model.fit(X_train,
          y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, y_test),
          callbacks=[checkpoint])

# Evaluate the model on the test data
score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', accuracy)
