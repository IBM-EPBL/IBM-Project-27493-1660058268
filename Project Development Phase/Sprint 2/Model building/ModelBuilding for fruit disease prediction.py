from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1)
x_train=train_datagen.flow_from_directory(r'C:\Users\devis\OneDrive\Desktop\Dataset Plant Disease\fruit-dataset\fruit-dataset\train',target_size=(128,128),batch_size=2,class_mode='categorical')

x_test=test_datagen.flow_from_directory(r'C:\Users\devis\OneDrive\Desktop\Dataset Plant Disease\fruit-dataset\fruit-dataset\test',target_size=(128,128),batch_size=2,class_mode='categorical')


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=40,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=70,kernel_initializer='random_uniform',activation='relu'))
model.add(Dense(units=6,kernel_initializer='random_uniform',activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=["accuracy"])
model.fit(x_train,steps_per_epoch=168,epochs=3,validation_data=x_test,validation_steps=52)



model.save(r'C:\Users\devis\OneDrive\Desktop\fruit.h5')
model.summary()
