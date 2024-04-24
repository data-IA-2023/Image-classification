# plot dog photos from the dogs vs cats dataset
from os import listdir
import os
# import torch
# os.environ["KERAS_BACKEND"] = "torch"
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.image import imread
from numpy import load
from os import makedirs
from shutil import copyfile
from random import seed
from random import random
import sys
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# create directories
def organize(*,labeldirs:list[str],subdirs:list[str],dataset_home:str,categories:list[str],val_ratio:float)->None:
    assert 0<=val_ratio<=1
    for subdir in subdirs:
        # create label subdirectories
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)
    # define ratio of pictures to use for validation
    # copy training dataset images into subdirectories
    src_directory = subdirs[0]
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = subdirs[1]
        if random() < val_ratio:
            dst_dir = subdirs[0]
        if file.startswith(categories[0]):
            dst = dataset_home + dst_dir + labeldirs[0]  + file
            if not os.path.exists(dst) : copyfile(src, dst)
        elif file.startswith(categories[1]):
            dst = dataset_home + dst_dir + labeldirs[1]  + file
            if not os.path.exists(dst) : copyfile(src, dst)



# define cnn model
# def define_model()->Sequential:
# 	model = Sequential()
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Dropout(0.2))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# compile model
# 	opt = SGD(learning_rate=0.001, momentum=0.9)
# 	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# 	return model
def define_model()->Model:
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history)->None:
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness(*,dataset_home:str,subdirs:list[str],path:str)->None:
	# define model
	model = define_model()
	datagen = ImageDataGenerator(featurewise_center=True)
 	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# create data generators
	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory(dataset_home+subdirs[0],
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory(dataset_home+subdirs[1],
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=2, verbose=1)
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
	model.save(path)

def load_model(*,path)->Model:
	model = define_model()
	model.load_weights(path)
	return model

def predict_class(*,model,image):
	prediction = model.predict(image)
	print(prediction)
	return prediction


def main()->None:
    subdirs = ['train/', 'test/']
    labeldirs = ['cats/', 'dogs/']
    dataset_home = 'dataset_dogs_vs_cats/'
    categories= ['cat', 'dog']
    organize(labeldirs=labeldirs,subdirs=subdirs,dataset_home=dataset_home,categories=categories,val_ratio=0.75)
    #run_test_harness(dataset_home=dataset_home,subdirs=subdirs,path='VGG16_cat_dog.h5')


if __name__=="__main__":
    main()