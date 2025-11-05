#imports
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
#print out versions of libaries
print("Tensorflow: ", tf.__version__ , " Seaborn:" , sns.__version__)

#load the data set
cifar10 = tf.keras.datasets.cifar10
#create the training and testing data frames
#(training images, training labels), (testing images, testing labels)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#show the number of examples in each labeled set

sns.countplot(x=y_train)
#show the plot!
plt.show()

#check to make sure there are NO values that are not a number (NaN)
print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing: ", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (32, 32, 3) #28x28 p0x, 1 color channel (grayscale) - 3 for RGB

#reshape the training and testing data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.astype('float32')/255.0 #normalize the data to be between 0 and 1
#same for testing
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test.astype('float32')/255.0 #normalize the data to be between 0 and 1
#convert our labels to be one-hot, not sparce
y_train = tf.one_hot(y_train.astype(np.int32), depth = 10)
y_test = tf.one_hot(y_test.astype(np.int32), depth = 10)

#show an example image from MNIST
plt.imshow(x_train[random.randint(0,59999)][:,:,0], cmap='grey')
plt.show()


batch_size = 128
num_classes = 10
epochs = 5

#build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (5,5), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ]
)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,epochs=10,validation_data=(x_test, y_test))


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color = 'b', label = 'Training Loss')
ax[0].plot(history.history['val_loss'], color = 'red', label = 'Validation Loss')
legend = ax[0].legend(loc = 'best', shadow = True)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(history.history['acc'], color = 'b', label = 'Training Accuracy')
ax[1].plot(history.history['val_acc'], color = 'red', label = 'Validation Accuracy')
legend = ax[1].legend(loc = 'best', shadow = True)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()