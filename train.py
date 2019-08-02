from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.layers import Conv1D, Flatten
from keras.layers import Bidirectional
from keras.utils import to_categorical
import glob
import wandb
from wandb.keras import WandbCallback
from tensorflow.python.client import device_lib

wandb.init()
config = wandb.config

config.max_len = 20#20#11
config.buckets = 40#40#20


# Cache pre-processed data
if len(glob.glob("*.npy")) == 0:
    save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 50
config.batch_size = 100

num_classes = 3

X_train = X_train.reshape(
    X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(
    X_test.shape[0], config.buckets, config.max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


# overide LSTM & GRU
if 'GPU' in str(device_lib.list_local_devices()):
    print("Using CUDA for RNN layers")
    LSTM = CuDNNLSTM
    GRU = CuDNNGRU
    
    
    

model = Sequential()

model.add(Conv2D(32,
                 (3,3),#(config.first_layer_conv_width, config.first_layer_conv_height),#(3,3)
                 input_shape=(config.buckets, config.max_len, 1),
                 activation='relu'))
model.add(MaxPooling2D())


model.add(Conv2D(32,
                 (3,3),#(config.first_layer_conv_width, config.first_layer_conv_height),#(3,3)
                 #input_shape=(40, 20, 1),
                 activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,
                 (1,1),#(config.first_layer_conv_width, config.first_layer_conv_height),#(3,3)
                 #input_shape=(40, 20, 1),
                 activation='relu'))


#model.add(GRU(100, return_sequences=True))
#model.add(GRU(config.hidden_dims))
#model.add(Bidirectional(LSTM(config.hidden_dims)))


model.add(Flatten(input_shape=(config.buckets, config.max_len, channels)))

#model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.3))

#recurrent nets
#convolution - need to do permute? time is first index (y axis)
#data augmentation - add noise


#https://app.wandb.ai/qualcomm/audio-aug2/runs/dfpeckhb/model
#model.add(Conv1D(250,#config.filters,
#                 3,#config.kernel_size,
#                 padding='valid',
#                 activation='relu'))



model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
config.total_params = model.count_params()


model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(
    X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
