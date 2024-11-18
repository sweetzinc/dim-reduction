#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from typing import Tuple, List, Dict, Optional
try :
    import models
except ImportError:
    import sys
    sys.path.append('/workspace/src')
    import models
from models.base_vae import BaseVAE
from models.encoders import ConvEncoder
from models.decoders import ConvDecoder
#%%
class VanillaVAE(BaseVAE):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int = 128,
        encoder_filters: List[int] = [32, 64, 64, 64],
        decoder_filters: Optional[List[int]] = None,
        name: str = "vanilla_vae",
        **kwargs
    ):
        super().__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            name=name,
            **kwargs
        )

        if not encoder_filters:
            raise ValueError("encoder_filters cannot be empty")

        # Default decoder filters if not provided
        if decoder_filters is None:
            decoder_filters = encoder_filters[::-1][1:] + [input_shape[-1]]
        else:
            decoder_filters = list(decoder_filters[:-1]) + [input_shape[-1]]

        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        # Calculate encoder output shape
        self.encoder_output_shape = self._calculate_encoder_output_shape(
            input_shape,
            encoder_filters,
            strides=2
        )

        # Initialize encoder and decoder
        self.encoder = ConvEncoder(
            latent_dim=latent_dim,
            conv_filters=encoder_filters
        )

        self.decoder = ConvDecoder(
            initial_input_shape=input_shape,
            encoder_output_shape=self.encoder_output_shape,
            conv_filters=decoder_filters,
            latent_dim=latent_dim
        )
    
    def _calculate_encoder_output_shape(
        self,
        input_shape: Tuple[int, int, int],
        encoder_filters: List[int],
        strides: int = 2
    ) -> Tuple[int, int, int]:
        """
        Calculate shape of the tensor after applying all encoder convolutions.
        
        Args:
            input_shape: Shape of the input images (height, width, channels).
            encoder_filters: List of filter counts for each encoder layer.
            strides: Strides for the convolution layers.
            
        Returns:
            Shape of the tensor after all encoder layers.
        """
        height, width, channels = input_shape
        for _ in encoder_filters:
            height = (height + strides - 1) // strides  # Ensure integer division rounds up
            width = (width + strides - 1) // strides  # Handle odd dimensions correctly
        return (height, width, encoder_filters[-1])  # Last filter count gives the depth

    
    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode input images to latent space.
        
        Args:
            x: Input images
            
        Returns:
            tuple: (mean, log_var) of the latent distribution
        """
        return self.encoder(x)
    
    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode latent vectors to images.
        
        Args:
            z: Latent vectors
            
        Returns:
            Reconstructed images
        """
        return self.decoder(z)
    
    def _compute_reconstruction_loss(self, data: tf.Tensor, reconstruction: tf.Tensor) -> tf.Tensor:
        """
        Compute binary crossentropy loss for image reconstruction.

        Args:
            data: Original input images.
            reconstruction: Reconstructed images.

        Returns:
            Reconstruction loss value.
        """
        # Flatten images
        flat_data = tf.reshape(data, [-1, tf.reduce_prod(self.input_shape)])  # (batch_size, num_features)
        flat_reconstruction = tf.reshape(reconstruction, [-1, tf.reduce_prod(self.input_shape)])  # (batch_size, num_features)
        
        # Binary crossentropy loss (already sums across features)
        loss_per_sample = tf.keras.losses.binary_crossentropy(flat_data, flat_reconstruction)  # (batch_size,)
        
        # Reduce across samples
        reconstruction_loss = tf.reduce_mean(loss_per_sample)  # Scalar
        
        return reconstruction_loss


def create_vae(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 128,
    encoder_filters: List[int] = [32, 64, 64, 64],
    decoder_filters: Optional[List[int]] = None,
    learning_rate: float = 1e-4
) -> VanillaVAE:
    """Create and compile a VAE model.
    
    Args:
        input_shape: Shape of input images
        latent_dim: Dimension of the latent space
        encoder_filters: List of filter numbers for encoder
        decoder_filters: Optional list of filter numbers for decoder
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled VAE model
    """
    vae = VanillaVAE(
        input_shape=input_shape,
        latent_dim=latent_dim,
        encoder_filters=encoder_filters,
        decoder_filters=decoder_filters
    )
    
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    return vae

#%%
if __name__ == "__main__":
    # Create and compile VAE
    vae = create_vae(input_shape=(28, 28, 1), 
                     latent_dim=128, 
                     encoder_filters=[32, 64, 64])
    dummy_input = tf.random.normal((1, 28, 28, 1))  # Replace with appropriate input dimensions
    vae(dummy_input)  # Run a forward pass to build the model
    vae.summary()
    
    # Test encoding and decoding
    x = tf.random.normal((11, 28, 28, 1))
    mean, log_var = vae.encode(x)
    z = vae.reparameterize(mean, log_var)
    print("x.shape=", x.shape)
    print("z.shape=", z.shape)
    print("vae.encoder_output_shape=", vae.encoder_output_shape)
    x_recon = vae.decode(z)
    print("x_recon.shape=", x_recon.shape)

#%%
if __name__ == "__main__":
    import numpy as np 
    epochs = 2
    batch_size = 32 
    # Load FashionMNIST dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    # Train the model
    vae.fit(x_train, epochs=epochs, batch_size=batch_size)


# %%
