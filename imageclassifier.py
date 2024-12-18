import imagerecognition
import tensorflow as tf
import keras
from keras import datasets, layers, models, Model
from keras import ResNet50
from keras import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from PIL import Image
import os


# Used the DeepModel Fashion Dataset: https://github.com/yumingj/DeepFashion-MultiModal/tree/main

class ImageClassifier :
    def __init__(self, img_height = 256, img_width = 256):
        self.img_height = img_height
        self.img_width = img_width
        self.base_model = ResNet50(
            include_top = False,
            weights = "imagenet", # Default pre-trained dataset from Keras
            input_tensor = None,
            input_shape = (img_height, img_width, 3),
            classes = 3
        )
    
    def load_dfds(self):
        image_dir = "/DeepFashion-MultiModal/images"
        parsing_dir = "/DeepFashion-MultiModal/segm"
        keypoints_dir = "/DeepFashion-MultiModal/keypoints/keypoints_loc.txt"
        labels_shape_dir = "/DeepFashion-MultiModal/labels/shape/shape_anno_all.txt"
        labels_fabric_dir = "/DeepFashion-MultiModal/labels/texture/fabric_ann.txt"
        labels_color_dir = "/DeepFashion-MultiModal/labels/texture/pattern_ann.txt"

        # Load all masks so we have all the tensors of each mask of every image
        keypoints = {} # Each array is 21 values long
        with open(keypoints_dir, 'r') as f:
            for line in f:
                parts = line.strip.split() # splits each line on spaces
                keypoints[parts[0]] = parts[1:] # Key for the dictionary is first element of each line, which is the image name, its attributes being everything after the first element
            
        shape = {} # Each array is 12 values long: <img_name> <shape_0> <shape_1> ... <shape_11>
        with open(labels_shape_dir, 'r') as f:
            for line in f:
                parts = line.strip.split() # splits each line on spaces
                shape[parts[0]] = parts[1:] # Key for the dictionary is first element of each line, which is the image name, its attributes being everything after the first element
            
        fabric = {} # Each array is 4 values long: <img_name> <upper_fabric> <lower_fabric> <outer_fabric>
        with open(labels_fabric_dir, 'r') as f:
            for line in f:
                parts = line.strip.split() # splits each line on spaces
                fabric[parts[0]] = parts[1:] # Key for the dictionary is first element of each line, which is the image name, its attributes being everything after the first element

        color = {} # Each array is 4 values long: <img_name> <upper_color> <lower_color> <outer_color>
        with open(labels_color_dir, 'r') as f:
            for line in f:
                parts = line.strip.split() # splits each line on spaces
                color[parts[0]] = parts[1:] # Key for the dictionary is first element of each line, which is the image name, its attributes being everything after the first element
           
        for image in os.listdir(image_dir): # Check all arrays are matching to their image correct
            clothing_type = np.unique(parsing_mask) # Tell us the type of clothing
            print("--------------------------------------------------")
            parsing_mask = np.array(Image.open(os.path.join(parsing_dir, image))) # Default parsing tensor, unique values tell us the clothing type
            print(f"Unique Parsing of {image}:", clothing_type.shape)
            print(f"Keypoints Shape of {image}:", keypoints[image])
            print(f"Shape Label of {image}:", shape[image])
            print(f"Fabric Label of {image}:", fabric[image])
            print(f"Color Label of {image}:", color[image])
            print("--------------------------------------------------")



            # Preprocess data
            # 1. Resize masks and images to a fixed size (e.g., 256x256).
            print("Resizing image and parsing mask...")
            print("--------------------------------------------------")
            image = np.array(Image.open(image)).resize((self.img_height, self.img.width))
            parsing_mask_resized = np.array(Image.open(parsing_mask)).resize((self.img_width, self.img_height), Image.NEAREST)
            print(f"Resized image shape:", image)
            print(f"Resized parsed mask: ", parsing_mask_resized)
            print("--------------------------------------------------")
            # 2. Normalize pixel values.
            image = image.astype('float32') / 255.0 # Normalize each image pixel to [0, 1]
            parsing_mask_resized = parsing_mask_resized.astype('float32') / 255.0  # Normalize mask to [0, 1]


    def train(self, image):
        # For parsing masks, train a semantic segmentation model (e.g., U-Net, DeepLabV3+).
        # For keypoints, use a regression model to predict keypoint coordinates.
        # For shape, fabric, and color labels, train classification models.
        pass
        
    def clothing_type(self, image):
        pass
        
    def classify_occasion(self, clothing_type):
        pass
    
    def classify_weather(self, clothing_type):
        pass

    def deploy(self):
        pass
