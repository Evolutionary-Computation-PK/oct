import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime

from params import MAX_EPOCHS
from preprocessing.DataLoaderFactory import DataLoaderFactory
from model.EfficientNet import EfficientNetOct
from training.ModelTrainer import ModelTrainer
from training.MetricsLogger import MetricsLogger
from training.utils import get_optimizer, FocalLoss

def main():
    # Load data
    train_images = np.load('dataset/train_images.npy')
    train_labels = np.load('dataset/train_labels.npy')
    val_images = np.load('dataset/val_images.npy')
    val_labels = np.load('dataset/val_labels.npy')
    test_images = np.load('dataset/test_images.npy')
    test_labels = np.load('dataset/test_labels.npy')
    
    # Load best parameters
    best_params = np.load('model/best_params.npy', allow_pickle=True).item()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize logger
    logger = MetricsLogger('runs/final_training')
    
    # Initialize the data loader factory with best parameters
    factory = DataLoaderFactory(batch_size=best_params['batch_size'])
    
    # Combine train and validation data for final training
    combined_images = np.concatenate([train_images, val_images])
    combined_labels = np.concatenate([train_labels, val_labels])
    
    # Create data loaders
    train_loader, val_loader = factory.get_loaders(train_images, train_labels, val_images, val_labels)
    final_train_loader, test_loader = factory.get_loaders(combined_images, combined_labels, test_images, test_labels)
    
    # Initialize model
    model = EfficientNetOct(
        num_classes=4,
        dense_units=best_params['dense_units'],
        dropout=best_params['dropout_rate']
    )
    model.freeze_base()
    model.to(device)
    
    # Initialize optimizer and loss
    optimizer = get_optimizer(
        best_params['optimizer'],
        model.parameters(),
        best_params['lr'],
        best_params['weight_decay']
    )
    criterion = FocalLoss(gamma=best_params['focal_loss_gamma'])
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Initialize trainer
    trainer = ModelTrainer(model, criterion, optimizer, device, logger)
    
    # Feature extraction phase
    print("Starting feature extraction phase...")
    trainer.train(
        train_loader, val_loader,
        epochs=best_params['feature_extraction_epochs'],
        phase='feature_extraction'
    )
    
    # Fine-tuning phase
    print("Starting fine-tuning phase...")
    model.unfreeze_top_layers(best_params['num_unfrozen'])
    optimizer = get_optimizer(
        best_params['optimizer'],
        model.parameters(),
        best_params['lr'] / 10,
        best_params['weight_decay']
    )
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
    
    # Train with early stopping
    best_recall, best_epoch, best_model_state = trainer.train_with_early_stopping(
        train_loader, val_loader, scheduler,
        max_epochs=MAX_EPOCHS, patience=best_params['patience'],
        phase='fine_tuning'
    )

    # TODO: Train model again on final_train_loader (train and val data) for best_epoch epoches

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Final validation on a test set
    print("\nFinal validation results on test set:")
    metrics = trainer.evaluate(test_loader)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        logger.log_metrics({metric_name: value}, 0, 'final_test')
    
    # Save the best model
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'best_epoch': best_epoch,
        'best_recall': best_recall,
        'hyperparameters': best_params,
        'test_metrics': metrics
    }, 'model/best_model.pt')
    
    print(f"\nBest model saved with disease recall: {best_recall:.4f}")
    logger.close()

if __name__ == '__main__':
    main()
