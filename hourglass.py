# Reference: 
import tensorflow as tf
from keras import *
from keras import Model, LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
import imageclassifier

learning_rate = 0.01
epochs = 150

def build_hourglass(backbone_output):
    pass