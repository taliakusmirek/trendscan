# Reference: https://medium.com/@abdualimov/unet-implementation-of-the-unet-architecture-on-tensorflow-for-segmentation-of-cell-nuclei-528b5b6e6ffd
import tensorflow as tf
from keras import *
from keras import Model, LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
import imageclassifier

learning_rate = 0.01
epochs = 150

def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def load_data():
    dataset = imageclassifier.load_dfds

def build_unet(backbone_output):
    pass