import tensorflow as tf
from typing import Tuple, List

class ConvEncoder(tf.keras.layers.Layer):
    """Convolutional encoder for VAE.
    
    Attributes:
        latent_dim: Dimension of the latent space
        conv_filters: List of numbers of filters for conv layers
        conv_kernel_size: Kernel size for conv layers
        conv_strides: Strides for conv layers
    """
    
    def __init__(
        self,
        latent_dim: int,
        conv_filters: List[int] = [32, 64, 64, 64],
        conv_kernel_size: int = 3,
        conv_strides: int = 2,
        name: str = "conv_encoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        
        # Conv layers
        self.conv_layers = []
        for filters in conv_filters:
            self.conv_layers.extend([
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=conv_kernel_size,
                    strides=conv_strides,
                    padding='same'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
            ]) 
        
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense layers for mean and log variance
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through encoder.
        
        Args:
            inputs: Input images
            training: Whether in training mode
            
        Returns:
            tuple: (mean, log_var) of the latent distribution
        """
        x = inputs
        # Pass through conv layers
        for layer in self.conv_layers:
            x = layer(x, training=training)
        x = self.flatten(x)
        
        # Get mean and log variance
        mean = self.dense_mean(x, training=training)
        log_var = self.dense_log_var(x, training=training)
        
        return mean, log_var
    
#%%
if __name__ == "__main__":
    # Define encoder
    encoder = ConvEncoder(latent_dim=128, conv_filters=[32,64,64])
    
    # Generate random image
    x = tf.random.normal((1, 28, 28, 5))

    # Pass through encoder
    mean, log_var = encoder(x)
    print(mean.shape, log_var.shape)
# %%
if __name__ == "__main__":
    # Define parameters for the encoder
    input_shape = (28, 28, 1)  # Example input shape
    latent_dim = 128
    conv_filters = [32, 64, 64]

    # Create an Input layer
    encoder_input = tf.keras.layers.Input(shape=input_shape)

    # Initialize the ConvEncoder
    encoder = ConvEncoder(latent_dim=latent_dim, conv_filters=conv_filters)

    # Connect the input to the ConvEncoder to create a Model
    mean, log_var = encoder(encoder_input)

    # Build a Model from the ConvEncoder
    encoder_model = tf.keras.Model(inputs=encoder_input, outputs=[mean, log_var], name="encoder")

    # Print the summary
    encoder_model.summary()