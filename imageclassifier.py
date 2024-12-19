import imagerecognition
import unet
import hourglass
import densenet
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

        # Create lists to store 
        self.loaded_images, self.loaded_parsing, self.loaded_keypoints, self.loaded_shape, self.loaded_fabric, self.loaded_color = []

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
            parsing_mask= parsing_mask_resized.astype('float32') / 255.0  # Normalize mask to [0, 1]

            # Add all values to their lists, and then turn the list into a numpy array
            self.loaded_images.append(image)
            self.loaded_parsing(parsing_mask)
            self.loaded_keypoints(keypoints[image])
            self.loaded_shape(shape[image])
            self.loaded_fabric(fabric[image])
            self.loaded_color(color[image])

            self.loaded_images = np.array(self.loaded_images)
            self.loaded_parsing = np.array(self.loaded_parsing)
            self.loaded_keypoints = np.array(self.loaded_keypoints)
            self.loaded_shape = np.array(self.loaded_shape)
            self.loaded_fabric = np.array(self.loaded_fabric)
            self.loaded_color = np.array(self.loaded_color)

            return { # Return dictionary of everything from the dataset, the keys being strings that are the feature labels
                'images' : self.loaded_images,
                'parsings' : self.loaded_parsing,
                'keypoints' : self.loaded_keypoints,
                'shapes' : self.loaded_shape,
                'fabrics' : self.loaded_fabric,
                'colors' : self.loaded_color,
            }


    def train(self, batch_size = 32, epochs = 300, learning_rate=0.01):
        # For parsing masks, train a semantic segmentation model (e.g., U-Net, DeepLabV3+).
        # For keypoints, use a regression model to predict keypoint coordinates.
        # For shape, fabric, and color labels, train classification models.

        # Split dataset into training and testing, generic 80-20 split
        # Note, data is preprocessed and normalized in the load function
        dataset = self.load_dfds()
        split = int(len(dataset["images"]) * 0.8)
        train_data = tf.data.Dataset.from_tensor_slices({ # Take 80% of dataset by querying dictionary via key which is string of corresponding feature
                'images' : dataset['images'][:split],
                'parsings' : dataset['parsings'][:split],
                'keypoints' : dataset['keypoints'][:split],
                'shapes' : dataset['shapes'][:split],
                'fabrics' : dataset['fabrics'][:split],
                'colors' : dataset['colors'][:split],
        })

        val_data = tf.data.Dataset.from_tensor_slices({ # Take 80% of dataset by querying dictionary via key which is string of corresponding feature
                'images' : dataset['images'][split:],
                'parsings' : dataset['parsings'][split:],
                'keypoints' : dataset['keypoints'][split:],
                'shapes' : dataset['shapes'][split:],
                'fabrics' : dataset['fabrics'][split:],
                'colors' : dataset['colors'][split:],
        })
        
        # Preprocess training and validation datasets
        train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Shuffle to prevent biased splits
        val_data = train_data.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        

        # Building model...



        # Backbone is resnet50
        backbone = self.base_model
        backbone_output = backbone.output

        # Parsing mask, u-net decoder with resnet50
        parsing_model = unet.build_unet(backbone_output)
        parsing_output = layers.Conv2D(1, activation="softmax") # Conv2D because we are predicting pixels
        
        # Keypoint mask, stacked hourglass network
        keypoint_model = hourglass.build_hourglass(backbone_output)
        keypoint_output = layers.Dense(42, activation="sigmoid") # Dense because we are predicting (x,y) coordinates, 42 numbers (21 keypoints, each having a x and y coordinate)
        
        # Multi(Shape, fabric, color), densenet121
        classifcation_model = densenet.build_densenet(backbone_output)
        shape_output = layers.Dense(12, activation="sigmoid")
        fabric_output = layers.Dense(4, activation="sigmoid")
        color_output = layers.Dense(4, activation="sigmoid")

        model = Model(
            inputs = backbone.input,
            outputs = [
                parsing_output,
                keypoint_output,
                shape_output,
                fabric_output,
                color_output,
            ]
        )

        # Define loss functions...
        loss_functions = {
            'parsing_output' : unet.dice_coefficient,
            'keypoint_output' : tf.keras.losses.MeanSquaredError(),
            'shape_output' : tf.keras.losses.BinaryCrossentropy(),
            'fabric_output' : tf.keras.losses.BinaryCrossentropy(),
            'color_output' : tf.keras.losses.BinaryCrossentropy(),
        }

        # Define loss weights...
        loss_weights = {
            'parsing_output' : 1.0,
            'keypoint_output' : 0.5,
            'shape_output' : 0.3,
            'fabric_output' : 0.3,
            'color_output' : 0.3,
        }

        # Compile model with loss functions
        model = model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss_functions,
            loss_weights= loss_weights,
            metrics = {
            'parsing_output' : ,
            'keypoint_output' : ,
            'shape_output' : ,
            'fabric_output' : ,
            'color_output' : ,
        })

        # Run, predict, see performance, via a epochs training loop
        # Referene: https://www.tensorflow.org/guide/core/quickstart_core



    def clothing_type(self, image):
        pass
        
    def classify_occasion(self, clothing_type):
        pass
    
    def classify_weather(self, clothing_type):
        pass

    def deploy(self):
        pass
