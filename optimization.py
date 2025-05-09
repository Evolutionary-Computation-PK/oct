import numpy as np
import torch
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from sklearn.model_selection import KFold

from params import N_TRIALS, N_STARTUP_TRIALS, N_WARMUP_STEPS, INTERVAL_STEPS, N_FOLDS, RANDOM_STATE, NUM_CLASSES, \
    MAX_EPOCHS
from preprocessing.DataLoaderFactory import DataLoaderFactory
from model.EfficientNet import EfficientNetOct
from training.ModelTrainer import ModelTrainer
from training.MetricsLogger import MetricsLogger
from training.utils import get_optimizer, FocalLoss

# Load data
train_images = np.load('dataset/train_images.npy')
train_labels = np.load('dataset/train_labels.npy')
val_images = np.load('dataset/val_images.npy')
val_labels = np.load('dataset/val_labels.npy')

# Debug prints
print("Train images shape:", train_images.shape)
print("Train images dtype:", train_images.dtype)
print("Sample min/max values:", np.min(train_images), np.max(train_images))

# Add debug prints for labels
print("\nLabel information:")
print("Train labels shape:", train_labels.shape)
print("Train labels dtype:", train_labels.dtype)
print("Train labels unique values:", np.unique(train_labels))
print("Sample train labels:", train_labels[:5])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 64)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    optimizer_name = trial.suggest_categorical('optimizer', ['RMSprop', 'Adam', 'AdamW'])
    num_unfrozen = trial.suggest_int('num_unfrozen', 0, 20)
    dense_units = trial.suggest_int('dense_units', 32, 512)
    gamma = trial.suggest_float('focal_loss_gamma', 1.0, 3.0)
    feature_extraction_epochs = trial.suggest_int('feature_extraction_epochs', 4, 12)
    patience = trial.suggest_int('patience', 5, 15)

    print(f"\n[Pipeline] Starting trial {trial.number} with params: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, dropout_rate={dropout_rate}, optimizer={optimizer_name}, num_unfrozen={num_unfrozen}, dense_units={dense_units}, gamma={gamma}, feature_extraction_epochs={feature_extraction_epochs}, patience={patience}")

    # Initialize logger for this trial
    logger = MetricsLogger(f'runs/trial_{trial.number}')

    # Initialize data loader factory
    factory = DataLoaderFactory(batch_size=batch_size)

    # Cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    print(f"[Pipeline] Starting cross-validation for trial {trial.number}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"[Pipeline] Starting fold {fold} for trial {trial.number}...")
        X_tr, X_val = train_images[train_idx], train_images[val_idx]
        y_tr, y_val = train_labels[train_idx], train_labels[val_idx]
        
        train_loader, val_loader = factory.get_loaders(X_tr, y_tr, X_val, y_val)
        
        # Initialize model
        model = EfficientNetOct(num_classes=NUM_CLASSES, dense_units=dense_units, dropout=dropout_rate)
        model.freeze_base()
        model.to(device)

        # Initialize optimizer and loss
        optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
        criterion = FocalLoss(gamma=gamma)
        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # Initialize trainer
        trainer = ModelTrainer(model, criterion, optimizer, device, logger)

        # Feature extraction phase
        print(f"[Pipeline] Trial {trial.number}, Fold {fold}: Starting feature extraction phase ({feature_extraction_epochs} epochs)...")
        trainer.train(
            train_loader, val_loader,
            epochs=feature_extraction_epochs,
            phase='feature_extraction',
            trial_num=trial.number,
            fold_num=fold
        )
        print(f"[Pipeline] Trial {trial.number}, Fold {fold}: Feature extraction phase complete.")
        
        # Fine-tuning phase with early stopping
        print(f"[Pipeline] Trial {trial.number}, Fold {fold}: Starting fine-tuning phase (max {MAX_EPOCHS} epochs, patience {patience})...")
        model.unfreeze_top_layers(num_unfrozen)
        optimizer = get_optimizer(optimizer_name, model.parameters(), lr / 10, weight_decay)
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
            max_epochs=MAX_EPOCHS, patience=patience,
            phase='fine_tuning',
            trial_num=trial.number,
            fold_num=fold
        )
        print(f"[Pipeline] Trial {trial.number}, Fold {fold}: Fine-tuning phase complete. Best recall: {best_recall}, Best epoch: {best_epoch}")
        fold_scores.append(best_recall)
        
        # Report intermediate value for pruning
        trial.report(np.mean(fold_scores), fold)
        
        # Handle pruning
        if trial.should_prune():
            print(f"[Pipeline] Trial {trial.number} pruned at fold {fold}.")
            logger.close()
            raise optuna.TrialPruned()
        print(f"[Pipeline] Trial {trial.number}, Fold {fold} complete. Current mean recall: {np.mean(fold_scores)}")

    logger.close()
    print(f"[Pipeline] Trial {trial.number} complete. Mean recall across folds: {np.mean(fold_scores)}")
    return float(np.mean(fold_scores))

def main():
    print("[Pipeline] Starting Optuna study...")
    # Initialize study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=N_STARTUP_TRIALS,
            n_warmup_steps=N_WARMUP_STEPS,
            interval_steps=INTERVAL_STEPS
        )
    )
    
    # Optimize
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Print results
    print('[Pipeline] Best trial:')
    trial = study.best_trial
    print(f'  Value (Disease Recall): {trial.value}')
    print('  Params:')
    for key, val in trial.params.items():
        print(f'    {key}: {val}')
    
    # Save best parameters
    best_params = trial.params
    np.save('model/best_params.npy', best_params)
    print('[Pipeline] Best parameters saved to model/best_params.npy')

if __name__ == '__main__':
    main()
