#%%
import tensorflow as tf
from typing import Tuple
import numpy as np 

def prepare_fashion_mnist(
    batch_size: int,
    buffer_size: int,
    validation_split: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepare FashionMNIST dataset for training.
    
    Args:
        batch_size: Batch size for training
        buffer_size: Buffer size for shuffling
        validation_split: Fraction of data to use for validation
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load FashionMNIST
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Combine train and test for splitting
    x_all = np.concatenate([x_train, x_test], axis=0)
    
    # Preprocess images
    x_all = x_all.astype('float32') / 255.0
    x_all = x_all[..., tf.newaxis]  # Add channel dimension
    
    # Calculate split index
    split_idx = int(len(x_all) * (1 - validation_split))
    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    
    # Configure datasets
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


#%% 
if __name__ == "__main__":
    train_data, val_data = prepare_fashion_mnist(batch_size=32, buffer_size=10000)
    print(train_data)
    print(val_data)
# %%
