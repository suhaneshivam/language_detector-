#simple feedforward network with three hidden layers and 6 output nodes for six language
#500,500,250,6 layers 
#all hidden layers have relu activation function
#output layers has a softmax activation function
#batch size:100
#no of epochs:4

from keras.models import Sequential
from keras.layers import Dense
from data_processing import *

#get training data

x=train_feat.drop('lang',axix=1)
y=encode(train_feat['lang'])

#define model
model=Sequential()
model.add(Dense(500,input_dim=663,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#train model
model.fit(x,y,epochs=4,batch_size=100)
