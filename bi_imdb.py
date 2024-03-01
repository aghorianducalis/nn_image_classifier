# Solving the applied problem of determining the tonality of a comment using recurrent neural networks
# bidirectional LSTM network processes the input sequence in both forward and backward directions.
# This allows the model to capture information from both past and future contexts,
# potentially improving performance, especially in tasks where context matters

from __future__ import print_function
import numpy as np
from keras import Input, Model
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.datasets import imdb
from numpy import concatenate

# Maximum number of words to consider as features
max_features = 20000
# Maximum length of sequences
max_length = 100

# Training parameters
batch_size = 30
nb_epoch = 8

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

# Convert y_train and y_test to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define input layer for the model
sequence_input = Input(shape=(max_length,), dtype='int32')

# Add Embedding layer to convert words into dense vectors of fixed size
embedded_sequences = Embedding(max_features, 128, input_length=max_length)(sequence_input)

# Bidirectional LSTM layers capture information from both past and future contexts
forwards_lstm = LSTM(128)(embedded_sequences)
backwards_lstm = LSTM(128, go_backwards=True)(embedded_sequences)

# Concatenate outputs of the two LSTM layers
merged = concatenate([forwards_lstm, backwards_lstm], axis=1)

# Apply dropout regularization to reduce overfitting
after_dropout = Dropout(0.5)(merged)

# Dense layer for classification
output = Dense(1, activation='sigmoid')(after_dropout)

print('Build model...')

# Define the model
model = Model(inputs=sequence_input, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

# Fit the model to the training data
model.fit(X_train,
          y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, y_test))

# Evaluate the model on the test data
score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', accuracy)
