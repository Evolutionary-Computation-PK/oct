import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Tuple, Dict, Any
import logging

from training.MetricsLogger import MetricsLogger
from training.utils import compute_metrics
from training.logger_config import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, logger: Optional[MetricsLogger] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics_logger = logger

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        logger.info(f"Starting training epoch with {total_batches} batches")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.debug(f"Batch {batch_idx}/{total_batches}, Current loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Training epoch completed. Average loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred, all_prob = [], [], []
        total_batches = len(val_loader)
        
        logger.info(f"Starting validation with {total_batches} batches")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                
                probs = outputs.cpu()
                preds = probs.argmax(dim=1)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(preds.tolist())
                all_prob.extend(probs.tolist())
                
                if batch_idx % 10 == 0:  # Log every 10 batches
                    logger.debug(f"Validation batch {batch_idx}/{total_batches}, Current loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_true, all_pred, all_prob)
        logger.info(f"Validation completed. Average loss: {avg_loss:.4f}, Metrics: {metrics}")
        return avg_loss, metrics

    def train_with_early_stopping(
        self,
        train_loader,
        val_loader,
        scheduler: ReduceLROnPlateau,
        max_epochs: int = 100,
        patience: int = 10,
        phase: str = '',
        trial_num: Optional[int] = None,
        fold_num: Optional[int] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Train model with early stopping and learning rate scheduling."""
        best_val_recall = 0
        best_epoch = 0
        no_improve_epochs = 0
        best_model_state = None
        
        logger.info(f"Starting training with early stopping. Max epochs: {max_epochs}, Patience: {patience}")
        
        for epoch in range(max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
            
            # Training step
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate_epoch(val_loader)
            val_recall = metrics['disease_recall']
            
            # Learning rate scheduling
            scheduler.step(val_recall)
            
            # Logging
            if self.metrics_logger is not None:
                self.metrics_logger.log_training_step(
                    train_loss, val_loss, metrics, epoch,
                    phase, trial_num, fold_num
                )
            
            # Early stopping check
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch = epoch
                no_improve_epochs = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best model found! Recall: {best_val_recall:.4f}")
            else:
                no_improve_epochs += 1
                logger.info(f"No improvement for {no_improve_epochs} epochs. Best recall: {best_val_recall:.4f}")
                
            if self.metrics_logger is not None:
                self.metrics_logger.log_best_epoch(best_epoch, epoch, phase, trial_num, fold_num)
                
            if no_improve_epochs >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch}')
                break
                
        return best_val_recall, best_epoch, best_model_state

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        phase: str = '',
        trial_num: Optional[int] = None,
        fold_num: Optional[int] = None
    ) -> None:
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate_epoch(val_loader)
            
            if self.metrics_logger is not None:
                self.metrics_logger.log_training_step(
                    train_loss, val_loss, metrics, epoch,
                    phase, trial_num, fold_num
                )
            
            logger.info(f"Epoch {epoch + 1}/{epochs} completed. "
                       f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                       f"Val f1: {metrics['f1']:.4f}")

    def evaluate(self, loader) -> Dict[str, float]:
        """Final evaluation without logging."""
        logger.info("Starting final evaluation")
        self.model.eval()
        all_true, all_pred, all_prob = [], [], []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                probs = outputs.cpu()
                preds = probs.argmax(dim=1)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(preds.tolist())
                all_prob.extend(probs.tolist())
                
        metrics = compute_metrics(all_true, all_pred, all_prob)
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics 