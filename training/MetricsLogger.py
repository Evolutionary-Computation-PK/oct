from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class MetricsLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    def log_metrics(self, metrics: dict, epoch: int, phase: str, trial_num: int = None, fold_num: int = None):
        """Log metrics to TensorBoard and print to stdout."""
        prefix_base = f'trial_{trial_num}/fold_{fold_num}/{phase}' if trial_num is not None else phase
        prefix = f'{prefix_base}_{self.timestamp}'
        
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{metric_name}', value, epoch)
            print(f"[MetricsLogger] Trial: {trial_num}, Fold: {fold_num}, Phase: {phase}, Epoch: {epoch}, Metric: {metric_name}, Value: {value}")

    def log_training_step(self, train_loss: float, val_loss: float, metrics: dict, epoch: int, 
                         phase: str, trial_num: int = None, fold_num: int = None):
        """Log complete training step metrics and print to stdout."""
        metrics['train_loss'] = train_loss
        metrics['val_loss'] = val_loss
        print(f"[MetricsLogger] Trial: {trial_num}, Fold: {fold_num}, Phase: {phase}, Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, Metrics: {metrics}")
        self.log_metrics(metrics, epoch, phase, trial_num, fold_num)

    def log_best_epoch(self, best_epoch: int, epoch: int, phase: str, trial_num: int = None, fold_num: int = None):
        """Log the best epoch number and print to stdout."""
        prefix_base = f'trial_{trial_num}/fold_{fold_num}/{phase}' if trial_num is not None else phase
        prefix = f'{prefix_base}_{self.timestamp}'
        self.writer.add_scalar(f'{prefix}/best_epoch', best_epoch, epoch)
        print(f"[MetricsLogger] Trial: {trial_num}, Fold: {fold_num}, Phase: {phase}, Epoch: {epoch}, Best epoch so far: {best_epoch}")

    def close(self):
        """Close the TensorBoard writer and print to stdout."""
        print("[MetricsLogger] Closing TensorBoard writer.")
        self.writer.close() 