#%%
import tensorflow as tf
from typing import Dict, Optional
from pathlib import Path
import yaml
import logging
from datetime import datetime
import sys 
sys.path.append('/workspace')

class VAETrainer:
    """Trainer class for VAE models."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.setup_logging()
        self.setup_model()
        self.setup_callbacks()
        
    def setup_logging(self):
        """Setup logging and experiment directories."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(self.config["logging"]["log_dir"]) / timestamp
        self.checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"]) / timestamp
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        
    def setup_model(self):
        """Setup VAE model based on configuration."""
        from src.models.vanilla_vae import create_vae
        
        self.model = create_vae(
            input_shape=tuple(self.config["model"]["input_shape"]),
            latent_dim=self.config["model"]["latent_dim"],
            encoder_filters=self.config["model"]["encoder_filters"],
            decoder_filters=self.config["model"]["decoder_filters"],
            learning_rate=self.config["training"]["learning_rate"]
        )
        
    def setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "weights-{epoch:02d}-{val_loss:.2f}.keras"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                save_freq=self.config["logging"]["save_freq"]
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config["training"]["early_stopping_patience"],
                restore_best_weights=True
            )
        ]
        
    def prepare_dataset(self, data_path: Optional[str] = None):
        """Prepare training and validation datasets.
        
        Args:
            data_path: Optional override for data directory path
        """
        # # This is a placeholder - implement your actual data loading logic
        # raise NotImplementedError("Implement data loading based on your specific needs")
        from src.data_handling.fashion_mnist import prepare_fashion_mnist
    
        self.train_dataset, self.val_dataset = prepare_fashion_mnist(
            batch_size=self.config["data"]["batch_size"],
            buffer_size=self.config["data"]["buffer_size"],
            validation_split=self.config["data"]["validation_split"]
        )
        
        logging.info(f"Prepared training dataset: {self.train_dataset}")
        logging.info(f"Prepared validation dataset: {self.val_dataset}")


    def train(self, train_data: tf.data.Dataset, val_data: Optional[tf.data.Dataset] = None):
        """Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
        """
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config["training"]["epochs"],
            callbacks=self.callbacks
        )
        
        # Save training history
        import json
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history.history, f)
            
        return history
    
#%%
if __name__ == "__main__":
    test_config = {
        "logging": {
            "log_dir": "logs",
            "checkpoint_dir": "checkpoints",
            "save_freq": 1
        },
        "model": {
            "input_shape": [64, 64, 3],
            "latent_dim": 128,
            "encoder_filters": [32, 64, 64, 64],
            "decoder_filters": None
        },
        "training": {
            "learning_rate": 1e-4,
            "epochs": 5,
            "early_stopping_patience": 3
        },
        "data": {
            "batch_size": 32,
            "buffer_size": 10000,
            "validation_split": 0.2
        }
    }
    
    with open("test_config.yaml", "w") as f:
        yaml.dump(test_config, f)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Create trainer instance
    trainer = VAETrainer("test_config.yaml")
    
    # Prepare datasets
    trainer.prepare_dataset()
    
    # Train model
    history = trainer.train(
        train_data=trainer.train_dataset,
        val_data=trainer.val_dataset
    )
    
    # Save final model
    final_model_path = trainer.checkpoint_dir / "final_model"
    trainer.model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

# %%
