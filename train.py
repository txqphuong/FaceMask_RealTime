from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D,InputLayer,LSTM
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np


image_directory='dataset\Test/'

face_mask=os.listdir(image_directory+ 'face_mask/')
face_only=os.listdir(image_directory+ 'face_only/')
dataset=[]
label=[]

INPUT_SIZE=64


for i , image_name in enumerate(face_mask):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'face_mask/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(face_only):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'face_only/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

print(x_train.shape)
y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=19, 
validation_data=(x_test, y_test),
shuffle=False)

model.save('model.h5')










