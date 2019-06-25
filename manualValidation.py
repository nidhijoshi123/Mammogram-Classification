# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:01:18 2019

@author: nidhi
"""

import shutil, os
import Path as Path
import cnn1 as cnn
#import Vgg as vgg
from sklearn.model_selection import train_test_split

cancerPath = Path.cancerPath
normalPath = Path.normalPath
cancerTrainPath = Path.cancerTrainPath
cancerValidationPath = Path.cancerValidationPath
normalTrainPath = Path.normalTrainPath
normalValidationPath = Path.normalValidationPath
trainPath = Path.trainPath
validationPath = Path.validationPath
trainCount = 0


#Split images into 80% train set and 20% test set randomly
def splitPath(path,trainPath,validationPath,rs):
    X = y = os.listdir(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    for x in X_train:
        if x!= 'desktop.ini':
            shutil.copy2(path+x , trainPath)
    for x in X_test:
        if x!= 'desktop.ini':
            shutil.copy2(path+x , validationPath)

#Count number of images in training set
def countTrainingSet():
    path, dirs, files = next(os.walk(cancerTrainPath))
    file_count_1 = len(files)
    path, dirs, files = next(os.walk(normalTrainPath))
    return len(files) + file_count_1
    
            
#Implement 5 fold cross validation
for rs in range(0,5):
    print ('I am in fold {} : '.format(rs+1 ) )
    if (os.path.isdir(trainPath) and os.path.isdir(Path.validationPath)):
        shutil.rmtree(trainPath)
        shutil.rmtree(validationPath)
        os.makedirs(cancerValidationPath)
        os.makedirs(cancerTrainPath)
        os.makedirs(normalValidationPath)
        os.makedirs(normalTrainPath)
    else:
        os.makedirs(cancerValidationPath)
        os.makedirs(cancerTrainPath)
        os.makedirs(normalValidationPath)
        os.makedirs(normalTrainPath)
        
        
    splitPath(cancerPath,cancerTrainPath,cancerValidationPath,rs)
    splitPath(normalPath,normalTrainPath,normalValidationPath,rs)
    cnn.fitCNN(countTrainingSet() )
    #vgg.fitVGG(countTrainingSet())
        
cnn.makePredictions('Validation confusion matrix :')      
        