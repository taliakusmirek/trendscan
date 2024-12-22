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
class DoubleConv(layers.Layer):
    def __init__(self, filters):
        super(DoubleConv, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding="same") # maintain dimension with padding = "same"
        self.bn1 = layers.BatchNormalization() # better stability
        self.conv2 = layers.Conv2D(filters, 3, padding="same") # maintain dimension with padding = "same"
        self.bn2 = layers.BatchNormalization() # better stability
        self.relu = layers.ReLU()
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class FashionNN :
    def __init__(self, num_parsing_classes, img_height = 256, img_width = 256):
        super(FashionNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.backbone = ResNet50(
            include_top = False,
            weights = "imagenet", # Default pre-trained dataset from Keras
            input_tensor = None,
            input_shape = (img_height, img_width, 3),
            classes = 3
        )
    
        for layer in self.backbone.layers[:100]:
            layer.trainable = False # freeze early layers of ResNet backbone
        
        self.densenet = tf.keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(img_height, img_width, 3)
        )

        for layer in self.densenet.layers[:100]:
            layer.trainable = False # freeze early layers of densenet backbone
        
        # Reference: https://arxiv.org/pdf/1505.04597

        # UNet decoder components, used for predicting parsing mask
        self.up_conv1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")
        self.double_conv1 = DoubleConv(512)
        self.up_conv2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")
        self.double_conv2 = DoubleConv(256)
        self.up_conv3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")
        self.double_conv3 = DoubleConv(128)
        self.up_conv4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")
        self.double_conv4 = DoubleConv(64)
        # final convolution layer for parsing
        self.final_conv = layers.Conv2D(num_parsing_classes, 1, activation='softmax')

        # classification heads for shape, fabric, and color of NN
        self.global_average = layers.GlobalAveragePooling2D()
        self.shape_dense = layers.Dense(12, activation="sigmoid") # Dense
        self.fabric_dense = layers.Dense(4, activation="sigmoid") # Dense
        self.color_dense = layers.Dense(4, activation="sigmoid") # Dense

    def build_unet_decoder(self, x, skip_connections):
        # First up-sampling block
        x = self.up_conv1(x)
        x = layers.Concatenate()([x, skip_connections[3]])
        x = self.double_conv1(x)
        
        # Second up-sampling block
        x = self.up_conv2(x)
        x = layers.Concatenate()([x, skip_connections[2]])
        x = self.double_conv2(x)
        
        # Third up-sampling block
        x = self.up_conv3(x)
        x = layers.Concatenate()([x, skip_connections[1]])
        x = self.double_conv3(x)
        
        # Fourth up-sampling block
        x = self.up_conv4(x)
        x = layers.Concatenate()([x, skip_connections[0]])
        x = self.double_conv4(x)
        
        return x

    def call(self, inputs):
        # get backbone and skip connections
        skip_layers = [
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'
        ]

        backbone_output = self.backbone(inputs)
        skip_connections = []
        for name in skip_layers:
            skip_connections.append(self.backbone.get_layer(name).output)
        
        unet_features = self.build_unet_decoder(backbone_output, skip_connections)
        parsing_output = self.final_conv(unet_features)
        densenet_output = self.densenet(inputs)
        pooled_features = self.global_average(densenet_output)
        shape_output = self.shape_dense(pooled_features)
        fabric_output = self.fabric_dense(pooled_features)
        color_output = self.color_dense(pooled_features)

        return {
            'parsing_output': parsing_output,
            'shape_output': shape_output,
            'fabric_output': fabric_output,
            'color_output': color_output
        }
    

    def load_dfds(self):
        image_dir = "/DeepFashion-MultiModal/images"
        parsing_dir = "/DeepFashion-MultiModal/segm"
        labels_shape_dir = "/DeepFashion-MultiModal/labels/shape/shape_anno_all.txt"
        labels_fabric_dir = "/DeepFashion-MultiModal/labels/texture/fabric_ann.txt"
        labels_color_dir = "/DeepFashion-MultiModal/labels/texture/pattern_ann.txt"

        # Create lists to store 
        self.loaded_images = [] 
        self.loaded_parsing = [] 
        self.loaded_shape = [] 
        self.loaded_fabric = []
        self.loaded_color = []

       
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
            print("--------------------------------------------------")
            parsing_mask = np.array(Image.open(os.path.join(parsing_dir, image))) # Default parsing tensor, unique values tell us the clothing type
            clothing_type = np.unique(parsing_mask) # Tell us the type of clothing
            print(f"Unique Parsing of {image}:", clothing_type.shape)
            print(f"Shape Label of {image}:", shape[image])
            print(f"Fabric Label of {image}:", fabric[image])
            print(f"Color Label of {image}:", color[image])
            print("--------------------------------------------------")

            # Preprocess data
            # 1. Resize masks and images to a fixed size (e.g., 256x256).
            print("Resizing image and parsing mask...")
            print("--------------------------------------------------")
            image = Image.open(os.path.join(image_dir, image))
            image = image.resize((self.img_height, self.img_width))
            image = np.array(image)
            parsing_mask_resized = Image.open(parsing_mask)
            parsing_mask_resized = parsing_mask_resized.resize((self.img_width, self.img_height), Image.NEAREST)
            print(f"Resized image shape:", image)
            print(f"Resized parsed mask: ", parsing_mask_resized)
            print("--------------------------------------------------")
            # 2. Normalize pixel values.
            image = image.astype('float32') / 255.0 # Normalize each image pixel to [0, 1]
            parsing_mask= parsing_mask_resized.astype('float32') / 255.0  # Normalize mask to [0, 1]

            # Add all values to their lists, and then turn the list into a numpy array
            self.loaded_images.append(image)
            self.loaded_parsing.append(parsing_mask)
            self.loaded_shape.append(shape[image])
            self.loaded_fabric.append(fabric[image])
            self.loaded_color.append(color[image])

        self.loaded_images = np.array(self.loaded_images)
        self.loaded_parsing = np.array(self.loaded_parsing)
        self.loaded_shape = np.array(self.loaded_shape)
        self.loaded_fabric = np.array(self.loaded_fabric)
        self.loaded_color = np.array(self.loaded_color)

        return { # Return dictionary of everything from the dataset, the keys being strings that are the feature labels
            'images' : self.loaded_images,
            'parsings' : self.loaded_parsing,
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
                'shapes' : dataset['shapes'][:split],
                'fabrics' : dataset['fabrics'][:split],
                'colors' : dataset['colors'][:split],
        })

        val_data = tf.data.Dataset.from_tensor_slices({ # Take 80% of dataset by querying dictionary via key which is string of corresponding feature
                'images' : dataset['images'][split:],
                'parsings' : dataset['parsings'][split:],
                'shapes' : dataset['shapes'][split:],
                'fabrics' : dataset['fabrics'][split:],
                'colors' : dataset['colors'][split:],
        })
        
        # Preprocess training and validation datasets
        train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Shuffle to prevent biased splits
        val_data = val_data.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Define training metrics...
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        parsing_accuracy = tf.keras.metrics.MeanIoU(num_classes=self.num_parsing_classes)
        shape_accuracy = tf.keras.metrics.BinaryAccuracy()
        fabric_accuracy = tf.keras.metrics.BinaryAccuracy()
        color_accuracy = tf.keras.metrics.BinaryAccuracy()

        # Define loss functions...
        loss_functions = {
            'parsing_output' : dice_loss,
            'shape_output' : tf.keras.losses.BinaryCrossentropy(),
            'fabric_output' : tf.keras.losses.BinaryCrossentropy(),
            'color_output' : tf.keras.losses.BinaryCrossentropy(),
        }

        # Define loss weights...
        loss_weights = {
            'parsing_output' : 1.0,
            'shape_output' : 0.3,
            'fabric_output' : 0.3,
            'color_output' : 0.3,
        }

        # Compile model with loss functions, loss weights, and training metrics
        model = model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss_functions,
            loss_weights= loss_weights,
            metrics = {
            'parsing_output' : [parsing_accuracy],
            'shape_output' : [shape_accuracy],
            'fabric_output' : [fabric_accuracy],
            'color_output' : [color_accuracy],
        })

        # Run, predict, see performance, via a epochs training loop
        # Referene: https://www.tensorflow.org/guide/core/quickstart_core
        def training(images, targets):
            # Calculate loss of each input
            with tf.GradientTape() as tape:
                predictions = self.call(images)
                # Calculate loss of each input
                parsing_loss = loss_functions['parsing_output'](targets['parsings'], predictions['parsing_output'])
                shape_loss = loss_functions['shape_output'](targets['shapes'], predictions['shape_output'])
                fabric_loss = loss_functions['fabric_output'](targets['fabrics'], predictions['fabric_output'])
                color_loss = loss_functions['color_output'](targets['colors'],predictions['color_output'])
                # Calculate total weight loss
                total_loss = ((loss_weights['parsing_output']*parsing_loss) +
                              (loss_weights['shape_output']*shape_loss) +
                              (loss_weights['fabric_output']*fabric_loss) +
                              (loss_weights['color_output']*color_loss))
                # Update weights based on loss of gradient descent iteration
                grads = tape.gradient(total_loss, self.trainable_variables)
                for g, v in zip(grads, self.trainable_variables):
                    v.assign_sub(learning_rate * g)
                # Update loss metrics, use update_state() to measure the metrics (mean, auc, accuracy) and stores them to be retrieved later
                train_loss.update_state(total_loss)
                parsing_accuracy.update_state(targets['parsings'], predictions['parsing_output'])
                shape_accuracy.update_state(targets['shapes'], predictions['shape_output'])
                fabric_accuracy.update_state(targets['fabrics'], predictions['fabric_output'])
                color_accuracy.update_state(targets['colors'], predictions['color_output'])

                return total_loss
            
        def validation_of_training(images, targets):
            predictions = self.call(images)

            # Calculate losses
            parsing_loss = loss_functions['parsing_output'](targets['parsings'], predictions['parsing_output'])
            shape_loss = loss_functions['shape_output'](targets['shapes'], predictions['shape_output'])
            fabric_loss = loss_functions['fabric_output'](targets['fabrics'], predictions['fabric_output'])  
            color_loss = loss_functions['color_output'](targets['colors'], predictions['color_output'])              
            
            total_loss = ((loss_weights['parsing_output']*parsing_loss) +
                              (loss_weights['shape_output']*shape_loss) +
                              (loss_weights['fabric_output']*fabric_loss) +
                              (loss_weights['color_output']*color_loss))
            
            val_loss.update_state(total_loss)
            return total_loss
        
        # The actual training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Reset metrics
            train_loss.reset_states()
            val_loss.reset_states()
            parsing_accuracy.reset_states()
            shape_accuracy.reset_states()
            fabric_accuracy.reset_states()
            color_accuracy.reset_states()

            # Training
            for batch in train_data:
                images = batch['images']
                targets = {
                    'parsings': batch['parsings'],
                    'shapes': batch['shapes'],
                    'fabrics': batch['fabrics'],
                    'colors': batch['colors']
                }
                training(images, targets)

            # Validation
            for batch in val_data:
                images = batch['images']
                targets = {
                    'parsings': batch['parsings'],
                    'shapes': batch['shapes'],
                    'fabrics': batch['fabrics'],
                    'colors': batch['colors']
                }
                validation_of_training(images, targets)

            # Metrics
            template = 'Epoch {}, Loss: {}, Val Loss: {}, Parsing IoU: {}, Shape Acc: {}, Fabric Acc: {}, Color Acc: {}'
            print(template.format(epoch + 1, train_loss.result(), val_loss.result(), parsing_accuracy.result(), shape_accuracy.result(), fabric_accuracy.result(), color_accuracy.result()))
        
            # Stop epochs if we find a maximizing/effective validation loss epoch
            if val_loss.result() < best_val_loss:
                best_val_loss = val_loss.result()
                self.save_weights('best_weights.h5')
            else:
                break

    def predict(self, image):
        image = image.resize((self.img_height, self.img_width))
        image = np.array(image).astype('float32')/255.0
        image = np.expand_dims(image, axis=0) # adds batch dimension

        predictions = self.call(image)
        parsing_mask = tf.argmax(predictions['parsing_output'][0], axis=-1)
        shape_pred = tf.round(predictions['shape_output'][0])
        fabric_pred = tf.round(predictions['fabric_output'][0])
        color_pred = tf.round(predictions['color_output'][0])
    
        predicted_clothing_type = np.unique(parsing_mask)

        return {
            'predicted clothing type': predicted_clothing_type.numpy(),
            'parsing_mask': parsing_mask.numpy(),
            'shape': shape_pred.numpy(),
            'fabric': fabric_pred.numpy(),
            'color': color_pred.numpy()
        }
    
    def model_evaluate_on_test(self, test_data):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        parsing_accuracy = tf.keras.metrics.MeanIoU(num_classes=self.num_parsing_classes)
        shape_accuracy = tf.keras.metrics.BinaryAccuracy()
        fabric_accuracy = tf.keras.metrics.BinaryAccuracy()
        color_accuracy = tf.keras.metrics.BinaryAccuracy()

        for batch in test_data:
            predictions = self.call(batch['images'])

            # Get accuracy metrics per batch on each feature
            parsing_accuracy.update_state(batch['parsings'], predictions['parsing_output'])
            shape_accuracy.update_state(batch['shapes'], predictions['shape_output'])
            fabric_accuracy.update_state(batch['fabrics'], predictions['fabric_output'])
            color_accuracy.update_state(batch['colors'], predictions['color_output'])
        
        return {
            'parsing_accuracy': parsing_accuracy.result().numpy(),
            'shape_accuracy': shape_accuracy.result().numpy(),
            'fabric_accuracy': fabric_accuracy.result().numpy(),
            'color_accuracy': color_accuracy.result().numpy()
        }
    
    def classify_occasion(self, clothing_type):
        occasion_mappings = {
            'formal': ['suit', 'dress', 'blazer', 'chinos', 'slacks'],
            'casual': ['t-shirt', 'jeans', 'sweater', 'skirt', 'shorts'],
            'sportswear': ['athletic_shorts', 'leggings', 'tank_top', 'tracksuit', 'tennis skirt'],
            'business_casual': ['khakis', 'polo', 'blouse', 'button-up']
        }
        suitable_occasions = []
        for occasion, clothes in occasion_mappings.items():
            for item in clothes:
                if item in clothing_type:
                    suitable_occasions.append(occasion)
        
        return suitable_occasions
    
    def classify_weather(self, clothing_type):
        weather_mappings = {
            'hot': ['t-shirt', 'shorts', 'tank_top', 'dress'],
            'cold': ['sweater', 'coat', 'jacket', 'scarf'],
            'moderate': ['blouse', 'light_jacket', 'jeans'],
            'rainy': ['raincoat', 'waterproof_jacket', 'boots']
        }
        suitable_weather = []
        for weather, clothes in weather_mappings.items():
            for item in clothes:
                if item in clothing_type:
                    suitable_weather.append(weather)
        return suitable_weather
    
    def deploy(self):
        pass



def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)



def main():
    num_parsing_classes = 21
    model = FashionNN(num_parsing_classes=num_parsing_classes)

    BATCH_SIZE = 32
    EPOCHS = 300
    LEARNING_RATE = 0.001

    # Train model
    try:
        print("Starting model training...")
        model.train(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    except Exception as e:
        print("Error during training: ", e)
        return
    
    # Test on dataset
    