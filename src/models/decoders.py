#%%
import tensorflow as tf
from typing import List, Tuple


class ConvDecoder(tf.keras.layers.Layer):
    """
    Convolutional decoder for VAE.

    Attributes:
        initial_input_shape: Shape of the original input to the VAE.
        conv_filters: List of filter sizes for each Conv2DTranspose layer.
        latent_dim: Dimension of the latent space.
        conv_kernel_size: Kernel size for Conv2DTranspose layers.
        conv_strides: Strides for Conv2DTranspose layers.
    """

    def __init__(
        self,
        initial_input_shape: Tuple[int, int, int],
        encoder_output_shape: Tuple[int, int, int],
        conv_filters: List[int],
        latent_dim: int,
        conv_kernel_size: int = 3,
        conv_strides: int = 2,
        name: str = "conv_decoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        if not conv_filters:
            raise ValueError("conv_filters cannot be empty")

        self.initial_input_shape = initial_input_shape
        self.encoder_output_shape = encoder_output_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides

        # Calculate conv transpose input shape
        self.convt_input_shape = self._calculate_convt_input_shape(
            initial_input_shape, conv_filters[:-1], conv_strides
        )

        # Calculate dense layer units from `convt_input_shape`
        self.units = int(tf.reduce_prod(self.convt_input_shape))

        # Dense layer to project latent space to match `convt_input_shape`
        self.dense_projector = tf.keras.layers.Dense(
            units=self.units, activation='relu'
        )

        # Deconv layers
        self.conv_transpose_layers = []
        for filters in conv_filters[:-1]:
            self.conv_transpose_layers.extend([
                tf.keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=conv_kernel_size,
                    strides=conv_strides,
                    padding='same'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
            ])

        # Final adjustment layer
        self.final_conv = tf.keras.layers.Conv2DTranspose(
            filters=conv_filters[-1],
            kernel_size=(3, 3),  # Adjust kernel size to fine-tune dimensions
            strides=1,
            padding='valid',  # Use 'valid' padding for precise control
            activation='sigmoid'
        )

    def _calculate_convt_input_shape(
        self, final_shape: Tuple[int, int, int], filters: List[int], strides: int
    ) -> Tuple[int, int, int]:
        """
        Calculate the starting shape for the Conv2DTranspose layers.

        Args:
            final_shape: Desired final output shape (e.g., (28, 28, 1)).
            filters: List of filters for the Conv2DTranspose layers.
            strides: Strides used in Conv2DTranspose layers.

        Returns:
            The starting shape for the Conv2DTranspose layers.
        """
        height, width, channels = final_shape
        for _ in filters:
            height = (height + strides - 1) // strides  # Reverse stride calculation
            width = (width + strides - 1) // strides
        return (height, width, filters[0])

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the decoder.

        Args:
            inputs: Latent space vectors.
            training: Whether in training mode.

        Returns:
            Reconstructed images.
        """
        # Project latent space to `convt_input_shape`
        x = self.dense_projector(inputs, training=training)
        x = tf.reshape(x, (-1, *self.convt_input_shape))

        # Pass through deconv layers
        for layer in self.conv_transpose_layers:
            x = layer(x, training=training)

        # Apply final adjustment layer
        x = self.final_conv(x)

        # Explicitly crop to ensure exact match
        x = tf.image.resize_with_crop_or_pad(x, self.initial_input_shape[0], self.initial_input_shape[1])

        return x


# Test the ConvDecoder
if __name__ == "__main__":
    decoder = ConvDecoder(
        initial_input_shape=(28, 28, 1),
        encoder_output_shape=(4, 4, 64),
        conv_filters=[64, 32, 1],
        latent_dim=128
    )

    z = tf.random.normal((11, 128))  # Batch size of 11
    x_recon = decoder(z)
    print(f"Output shape: {x_recon.shape}")  # Should be (11, 28, 28, 1)

    print("decooder.encoder_output_shape", decoder.encoder_output_shape)
    print("decoder.convt_input_shape", decoder.convt_input_shape)
# %%
