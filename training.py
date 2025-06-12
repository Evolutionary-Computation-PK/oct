import logging
import os
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from params import NUM_CLASSES, RANDOM_STATE, BEST_PARAMS, MAX_EPOCHS
from preprocessing.DataLoaderFactory import DataLoaderFactory
from model.EfficientNet import EfficientNetOct
from training.ModelTrainer import ModelTrainer
from training.MetricsLogger import MetricsLogger
from training.utils import get_optimizer, FocalLoss
from training.logger_config import setup_logger

# Setup main logger
logger = setup_logger('training', 'logs/training.log')

def train_model(model_idx: int, params: dict, train_images: np.ndarray, train_labels: np.ndarray,
                val_images: np.ndarray, val_labels: np.ndarray,
                test_images: np.ndarray, test_labels: np.ndarray, device: torch.device):
    """Train a single model with given parameters."""
    # Create model-specific directories
    model_dir = f'models/model_{model_idx}'
    results_dir = f'results/model_{model_idx}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup model-specific logger
    model_logger = setup_logger(f'training_model_{model_idx}', f'logs/model_{model_idx}.log')
    model_logger.info(f"Starting training for model {model_idx} with parameters: {params}")
    
    # Data loaders
    factory = DataLoaderFactory(batch_size=params['batch_size'])
    train_loader, val_loader = factory.get_loaders(train_images, train_labels, val_images, val_labels)
    test_dataset = factory.get_loaders(test_images, test_labels, test_images, test_labels)[1]

    # Initialize model
    model = EfficientNetOct(num_classes=NUM_CLASSES, 
                           dense_units=params['dense_units'], 
                           dropout=params['dropout_rate'])
    model.freeze_base()
    model.to(device)
    
    # Initialize optimizer and loss
    optimizer = get_optimizer(params['optimizer'], model.parameters(), 
                            params['lr'], params['weight_decay'])
    criterion = FocalLoss(gamma=params['focal_loss_gamma'])

    metrics_logger = MetricsLogger(f'runs/model_{model_idx}')

    trainer = ModelTrainer(model, criterion, optimizer, device, metrics_logger=metrics_logger)
    
    # Feature extraction phase
    model_logger.info(f"Starting feature extraction phase ({params['feature_extraction_epochs']} epochs)...")
    trainer.train(
        train_loader, val_loader,
        epochs=params['feature_extraction_epochs'],
        phase='feature_extraction'
    )
    
    # Fine-tuning phase
    model_logger.info(f"Starting fine-tuning phase...")
    model.unfreeze_top_layers(params['num_unfrozen'])
        
    optimizer = get_optimizer(params['optimizer'], model.parameters(), 
                            params['lr'] / 10, params['weight_decay'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        threshold=1e-3,
        threshold_mode='rel',
        cooldown=2,
        min_lr=1e-6
    )
    
    best_recall, best_epoch, best_model_state = trainer.train_with_early_stopping(
        train_loader, val_loader, scheduler,
        max_epochs=MAX_EPOCHS,
        patience=params['patience'],
        phase='fine_tuning'
    )
    
    # Save the best model
    torch.save(best_model_state, os.path.join(model_dir, 'weights.pt'))
    model_logger.info(f"Best model saved to {model_dir}/weights.pt")
    
    # Create a new model instance for evaluation
    eval_model = EfficientNetOct(num_classes=NUM_CLASSES, 
                               dense_units=params['dense_units'], 
                               dropout=params['dropout_rate'])
    eval_model.to(device)
    
    # Load best model and evaluate on test set
    eval_model.load_state_dict(best_model_state)
    test_metrics = trainer.evaluate(test_dataset)
    
    # Save test results
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    model_logger.info(f"Test metrics: {test_metrics}")
    metrics_logger.close()
    
    return test_metrics

def main():
    # Load data
    logger.info("Loading data...")
    train_images = np.load('dataset/train_images.npy')
    train_labels = np.load('dataset/train_labels.npy')
    val_images = np.load('dataset/val_images.npy')
    val_labels = np.load('dataset/val_labels.npy')
    test_images = np.load('dataset/test_images.npy')
    test_labels = np.load('dataset/test_labels.npy')
    
    # Uncomment for debugging with a smaller dataset
    train_images = train_images[:60000]
    train_labels = train_labels[:60000]
    val_images = val_images[:9000]
    val_labels = val_labels[:9000]
    # test_images = test_images[:48]
    # test_labels = test_labels[:48]
    
    # Debug prints
    logger.info(f"Train images shape: {train_images.shape}")
    logger.info(f"Val images shape: {val_images.shape}")
    logger.info(f"Test images shape: {test_images.shape}")
    logger.info(f"Train labels unique values: {np.unique(train_labels)}")
    
    # Setup device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache")
    
    # Train models
    all_test_metrics = []
    for i, params in enumerate(BEST_PARAMS):
        logger.info(f"\nTraining model {i+1}/{len(BEST_PARAMS)}")
        
        # Clear GPU memory before each model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info(f"Cleared GPU memory cache before training model {i+1}")
            
        test_metrics = train_model(i+1, params, train_images, train_labels,
                                 val_images, val_labels,
                                 test_images, test_labels, device)
        all_test_metrics.append(test_metrics)
    
    # Save combined results
    combined_results = {
        f'model_{i+1}': metrics 
        for i, metrics in enumerate(all_test_metrics)
    }
    
    with open('results/combined_test_metrics.json', 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    logger.info("Training completed. Results saved to results/combined_test_metrics.json")

if __name__ == '__main__':
    main()
