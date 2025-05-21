import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Tuple, Dict, Any

from training.MetricsLogger import MetricsLogger
from training.utils import compute_metrics


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, logger: Optional[MetricsLogger] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred, all_prob = [], [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                
                probs = outputs.cpu()
                preds = probs.argmax(dim=1)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(preds.tolist())
                all_prob.extend(probs.tolist())
                
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_true, all_pred, all_prob)
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
        
        for epoch in range(max_epochs):
            # Training step
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate_epoch(val_loader)
            val_recall = metrics['disease_recall']
            
            # Learning rate scheduling
            scheduler.step(val_recall)
            
            # Logging
            if self.logger is not None:
                self.logger.log_training_step(
                    train_loss, val_loss, metrics, epoch,
                    phase, trial_num, fold_num
                )
            
            # Early stopping check
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch = epoch
                no_improve_epochs = 0
                best_model_state = self.model.state_dict().copy()
            else:
                no_improve_epochs += 1
                
            if self.logger is not None:
                self.logger.log_best_epoch(best_epoch, epoch, phase, trial_num, fold_num)
                
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch}')
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
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate_epoch(val_loader)
            
            if self.logger is not None:
                self.logger.log_training_step(
                    train_loss, val_loss, metrics, epoch,
                    phase, trial_num, fold_num
                )
            
            print(f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}, "
                  f"Val f1={metrics['f1']:.4f}")

    def evaluate(self, loader) -> Dict[str, float]:
        """Final evaluation without logging."""
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
                
        return compute_metrics(all_true, all_pred, all_prob) 