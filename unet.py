import tensorflow as tf

def build_unet(backbone_output, output_channels):
    # Get intermediate layers from ResNet50 for skip connections
    layer_names = [
        'conv2_block3_out',   # 64x64
        'conv3_block4_out',   # 32x32
        'conv4_block6_out',   # 16x16
        'conv5_block3_out'    # 8x8
    ]
    
    # Helper function for upsampling blocks
    def upsample(filters, size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
    
    # Create upsampling stack
    up_stack = [
        upsample(512, 3),  # 8x8 -> 16x16
        upsample(256, 3),  # 16x16 -> 32x32
        upsample(128, 3),  # 32x32 -> 64x64
        upsample(64, 3),   # 64x64 -> 128x128
    ]
    
    # Get skip connections
    skips = [backbone.get_layer(name).output for name in layer_names]
    x = skips[-1]  # Start from the bottleneck
    skips = reversed(skips[:-1])  # Reverse remaining skip connections
    
    # Upsampling and establishing skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    # Final upsampling to match input size
    x = tf.keras.layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='softmax'
    )(x)
    
    return x

class FashionUNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(FashionUNet, self).__init__()
        
        # Initialize ResNet50 backbone
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(256, 256, 3)
        )
        
        # Freeze early layers
        for layer in self.backbone.layers[:100]:
            layer.trainable = False
            
        self.num_classes = num_classes
        
    def call(self, inputs):
        # Get backbone features
        x = self.backbone(inputs)
        
        # Build U-Net decoder
        segmentation_output = build_unet(x, self.num_classes)
        
        return segmentation_output

# Training setup
def train_unet(train_dataset, val_dataset):
    model = FashionUNet(num_classes=OUTPUT_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy', dice_coefficient]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_unet_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history