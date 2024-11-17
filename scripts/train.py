import argparse
from pathlib import Path
import tensorflow as tf
from src.training.trainer import VAETrainer

#TODO
def main():
    parser = argparse.ArgumentParser(description="Train VAE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-dir", type=str, help="Override data directory from config")
    args = parser.parse_args()
    
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Initialize trainer
    trainer = VAETrainer(args.config)
    
    # Load and prepare data
    train_data, val_data = trainer.prepare_dataset(args.data_dir)
    
    # Train model
    history = trainer.train(train_data, val_data)
    
if __name__ == "__main__":
    main()