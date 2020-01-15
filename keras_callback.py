import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import fashion_mnist
 
batch_size = 128
num_classes = 10
epochs = 50
 
# input image dimensions
img_rows, img_cols = 28, 28
 
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 
 
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
 
#Building our CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
#compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
 
#defining our callbacks
filepath = 'C:/Users/Niranjan/Desktop/weights/weights-{epoch:02d}-{val_acc:.2f}.hdf5'
early_stop = EarlyStopping(monitor='val_loss',patience=7, verbose=1, mode='auto')
model_ckpt = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, mode='auto',filepath=filepath)
csv_log = CSVLogger('training.log', append=False)
rlrp = ReduceLROnPlateau(monitor='val_loss',factor=0.1, mode='min', patience=5, verbose=1)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
 
def on_epoch_end(_,logs):
    THRESHOLD = 0.90
    if(logs['val_acc']> THRESHOLD):
        model.stop_training=True
stop_train = LambdaCallback(on_epoch_end=on_epoch_end)
 
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          callbacks=[early_stop, model_ckpt, csv_log, rlrp, tb, stop_train])
 
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])