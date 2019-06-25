# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,auc
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np
import Path as Path
import os

batch_size = 4
#from keras import applications


    
model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3),
 	                 activation='relu',
 	                 input_shape=(128,128,1)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
 
# =============================================================================
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Predicting test set images
def makePredictions(message):
    from keras.models import load_model
    model = load_model('best_model.h5')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(Path.testPath,
                                             target_size = (128, 128),
                                             color_mode="grayscale",
                                            batch_size = batch_size,
                                              class_mode = 'binary')
    y_p = []
    
    #desktop.ini files are hidden and are required to be removed.
    
    for file in os.listdir(Path.cancerTestPath):
        if file != 'desktop.ini':
            test_image = image.load_img(Path.cancerTestPath+file, target_size = (128, 128), grayscale=True)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            if result[0][0] == 1:
                y_p.append(1)
            else:
                y_p.append(0)
    
    for file in os.listdir(Path.normalTestPath):
        if file != 'desktop.ini':
            test_image = image.load_img(Path.normalTestPath+file, target_size = (128, 128), grayscale=True)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            if result[0][0] == 1:
                y_p.append(1)
            else:
                y_p.append(0)
     
    print(y_p)
    print(message)
    print(confusion_matrix(test_set.classes, y_p))
    
    target_names = ['Cancerous', 'Normal']
    print(classification_report(test_set.classes, y_p, target_names=target_names))
 
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set.classes, y_p)
    auc_keras = auc(fpr_keras, tpr_keras)
    print(auc_keras)
    
    
# Part 2 - Fitting the CNN to training set images
def fitCNN(trainCount):
    
    train_datagen = ImageDataGenerator(rescale = 1./255)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory(Path.trainPath,
                                                 target_size = (128, 128),
                                                  color_mode="grayscale",
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(Path.validationPath,
                                             target_size = (128, 128),
                                            color_mode="grayscale",
                                            batch_size = batch_size,
                                             class_mode = 'binary') #Was originally categorical


    csv_logger = CSVLogger('log.csv', append=True, separator=';')


    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss'), csv_logger]



    history=model.fit_generator(training_set,
                         steps_per_epoch = trainCount,
                         callbacks=callbacks,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = (Path.numberOfImages - trainCount))

    print(history.history.keys())
    classesTestSet = test_set.classes
    print(classesTestSet)
    y_p = []
    for file in os.listdir(Path.cancerValidationPath):
        if file != 'desktop.ini':
            test_image = image.load_img(Path.cancerValidationPath+file, target_size = (128, 128), grayscale=True)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            if result[0][0] == 1:
                y_p.append(1)
            else:
                y_p.append(0)
    
    for file in os.listdir(Path.normalValidationPath):
        if file != 'desktop.ini':
            test_image = image.load_img(Path.normalValidationPath+file, target_size = (128, 128), grayscale=True)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            if result[0][0] == 1:
                y_p.append(1)
            else:
                y_p.append(0)
    print(y_p)
    print('Validation confusion matrix : ')
    print(confusion_matrix(test_set.classes, y_p))
    
    target_names = ['Cancerous', 'Normal']
    print(classification_report(test_set.classes, y_p, target_names=target_names))
 
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set.classes, y_p)
    #Area under curve
    auc_keras = auc(fpr_keras, tpr_keras)
    print(auc_keras)   

  
