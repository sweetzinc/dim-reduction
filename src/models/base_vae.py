import tensorflow as tf
# import tensorflow_probability as tfp
from typing import Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod

class BaseVAE(tf.keras.Model, ABC):
    """Abstract base class for Variational Autoencoders.
    
    This class provides the basic structure and shared functionality for VAE implementations.
    Inheriting classes need to implement the encode, decode, and reparameterize methods.
    
    Attributes:
        latent_dim (int): Dimension of the latent space
        input_shape (tuple): Shape of the input data
        beta (float): KL divergence weight (for β-VAE variants)
        name (str): Name of the model
    """
    
    def __init__(
        self,
        latent_dim: int,
        input_shape: Tuple[int, ...],
        beta: float = 1.0,
        name: str = "base_vae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.beta = beta
        
        # Track losses for monitoring
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    
    @property
    def metrics(self):
        """Define metrics to be tracked during training."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @abstractmethod
    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode input data to latent mean and log variance.
        
        Args:
            x: Input tensor to be encoded
            
        Returns:
            tuple: (mean, log_var) of the latent space distribution
        """
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode latent vectors to reconstruction.
        
        Args:
            z: Latent vector to be decoded
            
        Returns:
            Reconstructed tensor in input space
        """
        raise NotImplementedError
    
    def reparameterize(self, mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """Perform reparameterization trick.
        
        Args:
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """Forward pass through the VAE.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            If training:
                tuple: (reconstruction, mean, log_var)
            Else:
                reconstruction
        """
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decode(z)
        
        if training:
            return reconstruction, mean, log_var
        return reconstruction
    
    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a training step.
        
        Args:
            data: Batch of training data
            
        Returns:
            dict: Dictionary of loss metrics
        """
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, mean, log_var = self(data, training=True)
            
            # Compute losses
            reconstruction_loss = self._compute_reconstruction_loss(data, reconstruction)
            kl_loss = self._compute_kl_loss(mean, log_var)
            total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a test step.
        
        Args:
            data: Batch of test data
            
        Returns:
            dict: Dictionary of loss metrics
        """
        reconstruction, mean, log_var = self(data, training=True)
        
        reconstruction_loss = self._compute_reconstruction_loss(data, reconstruction)
        kl_loss = self._compute_kl_loss(mean, log_var)
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def _compute_reconstruction_loss(self, data: tf.Tensor, reconstruction: tf.Tensor) -> tf.Tensor:
        """Compute reconstruction loss (to be implemented by child classes).
        
        Args:
            data: Original input data
            reconstruction: Reconstructed data
            
        Returns:
            Reconstruction loss value
        """
        raise NotImplementedError
    
    def _compute_kl_loss(self, mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """Compute KL divergence loss.
        
        Args:
            mean: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            KL divergence loss value
        """
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        )
        return kl_loss
    
    def sample(self, n_samples: int) -> tf.Tensor:
        """Generate samples from the learned distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(z)
    
    def reconstruct(self, data: tf.Tensor) -> tf.Tensor:
        """Reconstruct input data.
        
        Args:
            data: Input data to reconstruct
            
        Returns:
            Reconstructed data
        """
        mean, log_var = self.encode(data)
        z = self.reparameterize(mean, log_var)
        return self.decode(z)