from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics


CNN = pd.read_csv('kdd99.csv', header=None)

CNN.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(CNN, name):
    dummies = pd.get_dummies(CNN[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        CNN[dummy_name] = dummies[x]
    CNN.drop(name, axis=1, inplace=True)


encode_text_dummy(CNN, 'protocol_type')
encode_text_dummy(CNN, 'service')
encode_text_dummy(CNN, 'flag')
encode_text_dummy(CNN, 'land')
encode_text_dummy(CNN, 'logged_in')
encode_text_dummy(CNN, 'is_host_login')
encode_text_dummy(CNN, 'is_guest_login')
encode_text_dummy(CNN, 'outcome')


X = CNN.iloc[:,1:42]
Y = CNN.iloc[:,0]
C = CNN.iloc[:,0]
T = CNN.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
#cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
#cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="softmax"))

# define optimizer and objective, compile cnn

cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn1results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/cnn1results/cnntrainanalysis1.csv',separator=',', append=False)
cnn.fit(X_train, y_train, epochs=3,validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
cnn.save("results/cnn1results/cnn_model.hdf5")


cnn.load_weights("results/cnn1results/cnn_model.hdf5")


cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


#y_pred = cnn.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred , average="None")
#precision = precision_score(y_test, y_pred , average="None")
#f1 = f1_score(y_test, y_pred, average="None")
#np.savetxt('res/expected1.txt', y_test, fmt='%01d')
#np.savetxt('res/predicted1.txt', y_pred, fmt='%01d')
#
#print("confusion matrix")
#print("----------------------------------------------")
#print("accuracy")
#print("%.6f" %accuracy)
#print("racall")
#print("%.6f" %recall)
#print("precision")
#print("%.6f" %precision)
#print("f1score")
#print("%.6f" %f1)
#cm = metrics.confusion_matrix(y_test, y_pred)
print("==============================================")