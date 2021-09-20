import numpy as np
import tensorflow as tf
import PIL
from collections import Counter
from keras.utils import np_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import os

class Fer2013BasicGenerator():
    
    def __init__(self, csv_path,
                 batch_size=32,
                 input_size=(48, 48, 1),
        self.num_classes = 7 
        self.width= input_size[1] 
        self.height = input_size[0] 
        self.num_epochs = 50
        self.batch_size = batch_size
        self.num_features = 64
        self.data = pd.read_csv(csv_path)
        #split data into training, validation and test set
        self.data_train = self.data[self.data['Usage']=='Training'].copy()
        self.data_val   = self.data[self.data['Usage']=='PublicTest'].copy()
        self.data_test  = self.data[self.data['Usage']=='PrivateTest'].copy()
        print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(self.data_train.shape, self.data_val.shape, self.data_test.shape))

    def preprocess(self, df, dataName):
        df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
        data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1,self.width, self.height,1)/255.0   
        data_Y = np_utils.to_categorical(df['emotion'], self.num_classes)  
        print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
        return data_X, data_Y


    def get_generators(self):
        self.train_X, self.train_Y = self.preprocess(self.data_train, "train") #training data
        self.val_X, self.val_Y     = self.preprocess(self.data_val, "val") #validation data
        self.test_X, self.test_Y   = self.preprocess(self.data_test, "test") #test data
        
        # data generator
        self.data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)   

        self.trainflow = self.data_generator.flow(self.train_X, self.train_Y, 32) 
        return self.trainflow, self.val_X,self.val_Y,self.test_X, self.test_Y


    def get_classweights(self,normalize = False):
        counter = Counter(self.classes)                          
        max_val = float(max(counter.values()))       
        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 
       
        if normalize:
            factor=1.0/sum(class_weights.values())
            for k in class_weights:
                class_weights[k] = class_weights[k]*factor
        return class_weights