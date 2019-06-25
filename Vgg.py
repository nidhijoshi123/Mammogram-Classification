import keras
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


import Path as Path
batch_size = 4

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,auc


base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(64,64,3), pooling=None, classes=2)

x_model = base_model.output

x_model = GlobalAveragePooling2D()(x_model)

x_model = Dense(1024, activation='relu')(x_model)
x_model = Dropout(0.5, name='dropout_1')(x_model)

x_model = Dense(256, activation='relu')(x_model)
x_model = Dropout(0.5, name='dropout_2')(x_model)
predictions = Dense(2, activation='softmax', name='output_layer')(x_model)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    
    
def fitVGG(trainCount): 
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory(Path.trainPath,
                                                 target_size = (64, 64),
                                                 #color_mode="grayscale",
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')


    test_set = test_datagen.flow_from_directory(Path.validationPath,
                                             target_size = (64, 64),
                                            #color_mode="grayscale",
                                            batch_size = batch_size,
                                             class_mode = 'categorical')


    csv_logger = CSVLogger('log_vgg.csv', append=True, separator=';')


    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model_vgg.h5', monitor='val_loss', save_best_only=True), csv_logger]

    history=model.fit_generator(training_set,
                         steps_per_epoch = trainCount,
                         callbacks=callbacks,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = Path.numberOfImages-trainCount)

    print(history.history.keys())

    plt.figure()
    plt.plot(history.history['acc'], 'orange', label='Training accuracy')
    plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
    plt.plot(history.history['loss'], 'red', label='Training loss')
    plt.plot(history.history['val_loss'], 'green', label='Validation loss')
    plt.legend()
    plt.show()


    #Y_pred = model.predict_generator(test_set, (1292-trainCount) // batch_size+1)
    y_pred = model.predict_classes(test_set,(Path.numberOfImages-trainCount) // batch_size+1)
    #y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_set.classes, y_pred))
    print('Classification Report')
    target_names = ['Cancerous', 'Normal']
    print(classification_report(test_set.classes, y_pred, target_names=target_names))
 
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set.classes, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print(auc_keras)