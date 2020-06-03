from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras import metrics
import random



def model1():
  model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4),(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(units=random.randint(100,200), activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))




def model2():
        model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4),(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4)(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=random.randint(100,200),activation='relu'))
        model.add(Dense(units=random.randint(50,100),activation='relu'))
        model.add(Dense(units=1,activation='sigmoid'))





def model3():
        model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4),(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4),(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(filters=random.randint(30,40), 
                        kernel_size=random.choice((3,3),(4,4),(5,5)), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=random.randint(100,200),activation='relu'))
        model.add(Dense(units=random.randint(50,100),activation='relu'))
        model.add(Dense(units=random.randint(20,45),activation='relu'))
        model.add(Dense(units=1,activation='sigmoid'))






model = Sequential()

x=random.randint(1,3)


if x==1:
        model.add(model1())
elif x==2:
        model.add(model2())
else:
        model.add(model3())




 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model_history=model.fit(
        training_set,
        steps_per_epoch=800,
        epochs=5,
        validation_data=test_set,
        validation_steps=80)




print(max(model_history.history['val_accuracy']))
if (max(model_history.history['val_accuracy'])) > 0.80 :
    model.save('model.h5')



accuracy_file = open('/root/Task3/accuracy.txt','w+')
accuracy_file.write (str(model_history.history['val_accuracy']))
accuracy_file.close()


                   