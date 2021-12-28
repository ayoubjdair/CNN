#Imports

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
#This will change our learning rate at the end of each EPOCH
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy

#Constants

BATCH_SIZE = 64   #20 32 128 256
EPOCHS = 5
NUM_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
KERNEL_SIZE = 3
STRIDES = 1
NUM_FILTERS = 16
IMG_CHANS = 3



#Getting the data

def load_data(display):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  samples = 3
  num_row = 3
  num_col = 10

  if display:
    print("Loading Data...")
    print()
    print('Cifar 10 Dataset:')
    print('X_train: ' + str(x_train.shape))
    print(x_train.shape[0], 'training samples')
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print(x_test.shape[0], 'testing samples')
    print('Y_test:  '  + str(y_test.shape))
    print()

    # get a segment of the dataset
    num = num_row*num_col
    images = x_train[:num]
    labels = y_train[:num]

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    print("Some samples from the Cifar 10 dataset:")
    print()
    plt.show()

  return x_train, y_train, x_test, y_test

display_data = load_data(True)
  

#Noramlise Data
Function to perform normalisation on data

def normalise(x_train, y_train, x_test, y_test):
  # Normalize data.
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train = x_train /255.
  x_test = x_test /255.

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  print("Normalised Data")
  print("x_train: " + str(x_train.shape))
  print("x_test: " + str(x_test.shape))

  return x_train, y_train, x_test, y_test

#Data Augmentation
Fcuntion to preform various data augmentation on our data

def augment_data(x_train):
  datagen = ImageDataGenerator(
      # divide inputs by std of dataset
      featurewise_std_normalization=False,
      # divide each input by its std
      samplewise_std_normalization=False,
      # randomly rotate images in the range (deg 0 to 180)
      rotation_range=0,
      # randomly shift images horizontally
      width_shift_range=0.1,
      # randomly shift images vertically
      height_shift_range=0.1,
      # set range for random shear
      shear_range=0.,
      # set range for random zoom
      zoom_range=0.,
      # set range for random channel shifts
      channel_shift_range=0.,
      # randomly flip images
      horizontal_flip=True,
      # randomly flip images
      vertical_flip=False,
      rescale=None,
      preprocessing_function=None,
      data_format=None,
      # set validation split
      validation_split=VALIDATION_SPLIT)
  
  datagen.fit(x_train)

  return datagen

#Adaptive Learning rate function
This will reduce the learning rate as the EPOCHS increase as described in the ResNet paper, more on this in our report

Our learning_rate starts at 0.1 and devided by 10

def adaptive_lr(EPOCHS):

    learning_rate = 0.1

    if EPOCHS > 4:
        learning_rate = learning_rate/10
    elif EPOCHS > 3:
        learning_rate = learning_rate/10
    elif EPOCHS > 2:
        learning_rate = learning_rate/10
    elif EPOCHS > 1:
        learning_rate = learning_rate/10
        
    print('Learning rate adapted: ', learning_rate)

    return learning_rate

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(adaptive_lr)
#This reduces the learning rate on plataeu
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
CALLBACKS = [lr_reducer, lr_scheduler]

#Top 1 & top 5 Accuracy functions

def top_5_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5) 

def top_1_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1) 

#L2 Regularisation function
This function will loop over all the layers of the ResNet-50 pretrained model, check if they are compatible with regularisation, and apply a reguliser from the Keras library.

def regularise(REG, RES):

  reg = REG
  res = RES

  for layer in res.layers:
      for attr in ['kernel_regularizer']:
          if hasattr(layer, attr):
            setattr(layer, attr, reg)
  return res

#ResNet-50 Architecture block
This function returns resnet layer to add to our model below using pretrained weights form the ImageNet dataset

def Residual_block():
  # weights='imagenet' = we use the ImageNet pre-trained weights
  # include_top=False = we remove the classification layer
  # input_shape=(256, 256, 3) = we define the input shape
  res = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
  # res.summary()
  return res

#Building Our Network
We add our classification layer on the top of the pre-trained ResNet50 model

Activations used are Relu for two layers and Softmax for the output

Dropout rate set to 50%

See report for detailed specifics

RES = Residual_block()
RES_REG = regularise(l2, RES)

def build():

  model = models.Sequential()
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.UpSampling2D((2,2)))
  model.add(layers.UpSampling2D((2,2)))
  model.add(RES_REG)
  model.add(layers.Flatten())
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(128, activation='relu', kernel_regularizer = 'l2'))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(64, activation='relu', kernel_regularizer = 'l2'))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10, activation='softmax'))

  return model

#Getting things started
Here we load the data, normalise it, & apply augmentation

x_train, y_train, x_test, y_test = load_data(False)
x_train, y_train, x_test, y_test = normalise(x_train, y_train, x_test, y_test)

DATAGEN = augment_data(x_train)

INPUT_SHAPE = x_train.shape[1:]
print("input shape: " + str(INPUT_SHAPE))


#Training the model

Finally we fit the data to our network, apply our LOSS and optimiser functions (see report for details) and define the metrics we want i.e accuracy, top 5, and top 1 accuracy

LOSS = 'binary_crossentropy'
OPTIMISER = Adam(learning_rate=adaptive_lr(0))
model = build()
model.compile(loss=LOSS,
              optimizer=OPTIMISER,
              metrics = ['accuracy', top_5_categorical_accuracy, top_1_categorical_accuracy])

# Fit the model on the batches generated by datagen.flow().
history = model.fit(DATAGEN.flow(x_train, y_train, 
                    batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS, verbose=VERBOSE, workers=1,
                    callbacks=CALLBACKS, 
                    use_multiprocessing=False)

model.summary()

#Evaluation
Below are plots showcasing our results

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(history.params)

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,(EPOCHS+1))
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=1)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()