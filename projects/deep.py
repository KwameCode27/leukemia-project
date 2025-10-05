# first neural network with keras tutorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the dataset
df = pd.read_csv('merged.csv' , encoding_errors= 'replace')
# split into input (X) and output (y) variables
dataset = df.values
X = dataset[:,0:17]
Y = dataset[:,18]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.5)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# define the keras model with relu activation mode and batch_size 32
model = Sequential()
model.add(Dense(12, input_shape=(17,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=32, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))

#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for loss mean_squared_error ', fontsize=20 )
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize= 20)
plt.title('feedforward with mean_squared_error loss',fontsize=20)
plt.show()

# define the keras model with relu activation mode and batch_size 32
model = Sequential()
model.add(Dense(12, input_shape=(17,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=32, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))

#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for binary_crossentropy', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('feedforward with binary_crossentropy loss',fontsize=20)
plt.show()
# define model for softmax with binary_crossentropy loss
model = Sequential()
model.add(Dense(12, input_shape=(17,), activation='softmax'))
model.add(Dense(8, activation='softmax'))
model.add(Dense(1, activation='softmax'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=10, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=200,
          validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for binary_crossentropy loss', )
plt.ylabel('Loss', fontsize=17)
plt.xlabel('Epoch', fontsize=17)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

#calculating confusion_matrix
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('feedforward with binary_crossentropy',fontsize=20)
plt.show()

## define model for relu with mean_squared_error loss
model = Sequential()
model.add(Dense(12, input_shape=(17,), activation='softmax'))
model.add(Dense(8, activation='softmax'))
model.add(Dense(1, activation='softmax'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=10, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=200,
          validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for mean_squared_error loss', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)
#calculating confusion_matrix
model.fit(X_train, Y_train, epochs=100, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction', fontsize=20)
plt.xlabel('Actual', fontsize=20)
plt.title('feedforward with mean_squared_error', fontsize=20)
plt.show()

#define Keras RNN with relu activation function and mean_squared_error loss

model = Sequential()
#Adding the first RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True, input_shape= (X_train.shape[1],1)))
model.add(Dropout(0.2))

#Adding the second RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#Adding the third RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#Adding the fourth RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50))
model.add(Dropout(0.2))

#Adding the output layer
model.add(Dense(units = 1))
#Compile the RNN
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#Fitting the RNN to the Training set
model.fit(X_train, Y_train, epochs=100, batch_size=32)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))
#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=100, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for loss mean_squared_error', fontsize=20)
plt.ylabel('Loss',fontsize=20 )
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('RNN with mean_squared_error',fontsize=20)
plt.show()

# define the keras model with relu activation mode with binary_crossentropy loss and batch_size 32
#Initialize RNN:
model = Sequential()
#Adding the first RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True, input_shape= (X_train.shape[1],1)))
model.add(Dropout(0.2))

#Adding the second RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#Adding the third RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#Adding the fourth RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50))
model.add(Dropout(0.2))
#Adding the output layer
model.add(Dense(units = 1))
#Compile the RNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the RNN to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))
#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for loss binary_crossentropy',fontsize=20 )
plt.ylabel('Loss',fontsize=20 )
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
sns.set(font_scale=1.4)
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('RNN with binary_crossentropy',fontsize=20)
plt.show()
# define the keras model with softmax activation mode with mean_squared_error loss and batch_size 32
#Initialize RNN:

model = Sequential()

#Adding the first RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True, input_shape= (X_train.shape[1],1)))
model.add(Dropout(0.2))

#Adding the second RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True))
model.add(Dropout(0.2))

#Adding the third RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True))
model.add(Dropout(0.2))

#Adding the fourth RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50))
model.add(Dropout(0.2))

#Adding the output layer
model.add(Dense(units = 1))
#Compile the RNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the RNN to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))
#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for loss binary_crossentropy',fontsize=20 )
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('RNN with binary_crossentropy',fontsize=20)
plt.show()

# define the keras model with softmax activation mode with mean_squared_error loss and batch_size 32
#Initialize RNN:

model = Sequential()

#Adding the first RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True, input_shape= (X_train.shape[1],1)))
model.add(Dropout(0.2))

#Adding the second RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True))
model.add(Dropout(0.2))

#Adding the third RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50, activation='softmax', return_sequences=True))
model.add(Dropout(0.2))

#Adding the fourth RNN layer and some Dropout regularization
model.add(SimpleRNN(units = 50))
model.add(Dropout(0.2))

#Adding the output layer
model.add(Dense(units = 1))
#Compile the RNN
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#Fitting the RNN to the Training set
model.fit(X_train, Y_train, epochs=100, batch_size=32)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))
#Fitting the to the Training set
model.fit(X_train, Y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_val, Y_val))
print(model.evaluate(X_test, Y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model for loss mean_squared_error',fontsize=20 )
plt.ylabel('Loss',fontsize=20 )
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test, Y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, Y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
sns.set(font_scale=1.4)
cm = confusion_matrix(Y_test, y_pred)
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('RNN with mean_squared_error',fontsize=20)
plt.show()

import json

# Collect the important results
metrics = {
    "final_model": {
        "train_loss": float(results1[0]),
        "train_accuracy": float(results1[1]),
        "test_loss": float(results[0]),
        "test_accuracy": float(results[1]),
        "confusion_matrix": cm.tolist()
    }
}

# Save results into deep_metrics.json file
with open("deep_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Metrics have been saved to deep_metrics.json")




