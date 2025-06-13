import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
import cv2
from torchvision import transforms

from params import NUM_CLASSES, BEST_PARAMS
from model.EfficientNet import EfficientNetOct
from training.logger_config import setup_logger
from preprocessing.DataLoaderFactory import DataLoaderFactory
from preprocessing.Preprocessor import Preprocessor

logger = setup_logger('validation', 'logs/validation.log')

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Use the last convolutional layer from features
        self.target_layer = model.base_model.features[-1]
        logger.info(f"Using layer {self.target_layer} for Grad-CAM")
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, target_class=None):
        output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        
        # Handle target_class whether it's a tensor or integer
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weighted combination of feature maps using broadcasting
        cam = torch.sum(weights[0][:, None, None] * activations[0], dim=0)
        
        # ReLU and normalization
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.numpy()

def load_data():
    """Load test data."""
    logger.info("Loading test data...")
    test_images = np.load('dataset/test_images.npy')
    test_labels = np.load('dataset/test_labels.npy')
    
    logger.info(f"Test images shape: {test_images.shape}")
    logger.info(f"Test images dtype: {test_images.dtype}")
    logger.info(f"Sample min/max values: {np.min(test_images)}, {np.max(test_images)}")
    logger.info(f"Test labels unique values: {np.unique(test_labels)}")
    
    return test_images, test_labels

def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(cm, model_idx, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Model {model_idx}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def find_example_images(preds, labels, probs, images, model_idx):
    """Find and save example images for each class."""
    save_dir = f'results/model_{model_idx}/examples'
    os.makedirs(save_dir, exist_ok=True)
    preprocessor = Preprocessor()
    
    # Save original and preprocessed images for each class
    for class_idx in range(NUM_CLASSES):
        # Get indices for current class
        class_indices = np.nonzero(labels == class_idx)[0]
        
        if len(class_indices) == 0:
            continue
            
        # Get probabilities for current class
        class_probs = probs[class_indices, class_idx]
        
        # Find best, medium, and worst predictions
        sorted_indices = np.argsort(class_probs)
        best_idx = class_indices[sorted_indices[-1]]
        medium_idx = class_indices[sorted_indices[len(sorted_indices)//2]]
        worst_idx = class_indices[sorted_indices[0]]
        
        # Process each example
        for idx, name in [(best_idx, 'best'), (medium_idx, 'medium'), (worst_idx, 'worst')]:
            # Save original image
            plt.figure(figsize=(8, 8))
            plt.imshow(images[idx], cmap='gray')
            plt.title(f'Class {class_idx} - {name}\nPred: {preds[idx]} (Prob: {probs[idx, preds[idx]]:.2f})')
            plt.axis('off')
            plt.savefig(f'{save_dir}/class_{class_idx}_{name}.jpg')
            plt.close()
            
            # Preprocess and save preprocessed image
            try:
                preprocessed_img = preprocessor.preprocess(images[idx])
                # Save as numpy array
                np.save(f'{save_dir}/class_{class_idx}_{name}_preprocessed.npy', preprocessed_img.numpy())
                logger.info(f"Saved preprocessed image for class {class_idx} - {name}")
            except Exception as e:
                logger.error(f"Error preprocessing image for class {class_idx} - {name}: {str(e)}")
                continue

def generate_gradcam(model, model_idx, device):
    """Generate Grad-CAM visualizations for example images."""
    grad_cam = GradCAM(model)
    
    save_dir = f'results/model_{model_idx}/gradcam'
    cam_dir = f'results/model_{model_idx}/cam_matrices'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    examples_dir = f'results/model_{model_idx}/examples'
    for img_file in os.listdir(examples_dir):
        if not img_file.endswith('_preprocessed.npy'):
            continue
            
        img_path = os.path.join(examples_dir, img_file)
        img_np = np.load(img_path).astype(np.float32)
        
        # Convert grayscale to RGB by repeating the channel
        if len(img_np.shape) == 2:  # If grayscale
            img_np = np.stack([img_np] * 3, axis=0)  # Convert to RGB (C, H, W)
        
        img_tensor = torch.from_numpy(img_np).float()
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        
        try:
            target_class = int(img_file.split('_')[1])
            if not (0 <= target_class < NUM_CLASSES):
                logger.warning(f"Invalid target class {target_class} for {img_file}, skipping...")
                continue
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not extract target class from {img_file}, skipping...")
            continue
        
        model.eval()
        try:
            cam = grad_cam(img_tensor, target_class=target_class)
            
            # Save CAM matrix
            cam_path = os.path.join(cam_dir, f'cam_{img_file}')
            np.save(cam_path, cam.astype(np.float32))
            
            # Resize CAM to match image size
            cam = cv2.resize(cam, (img_np.shape[2], img_np.shape[1]))
            
            # Ensure CAM values are in [0, 1] range
            cam = np.clip(cam, 0, 1)
            
            # Create heatmap using OpenCV
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            heatmap = np.float32(heatmap) / 255
            
            # Denormalize image for visualization
            img_np = np.transpose(img_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            img_np = img_np * std + mean  # Denormalize
            img_np = np.clip(img_np, 0, 1)  # Clip to valid range
            
            # Create visualization with increased contrast
            output = heatmap * 0.7 + img_np * 0.3  # Increase heatmap visibility
            output = np.clip(output, 0, 1)  # Ensure final output is in valid range
            
            # Save visualization
            output_path = os.path.join(save_dir, f'gradcam_{img_file.replace("_preprocessed.npy", ".jpg")}')
            plt.figure(figsize=(8, 8))
            plt.imshow(output)
            plt.axis('off')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Generated Grad-CAM for {img_file} with target class {target_class}")
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM for {img_file}: {str(e)}")
            continue
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_images, test_labels = load_data()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Evaluate each model
    all_metrics = {}
    
    for model_idx, params in enumerate(BEST_PARAMS, 1):
        logger.info(f"\nEvaluating model {model_idx}")
        
        # Initialize model
        model = EfficientNetOct(
            num_classes=NUM_CLASSES,
            dense_units=params['dense_units'],
            dropout=params['dropout_rate']
        )
        model.to(device)
        
        # Load weights
        weights_path = f'models/model_{model_idx}/weights.pt'
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
        # Create data loader using the same factory as in training
        factory = DataLoaderFactory(batch_size=params['batch_size'])
        _, test_loader = factory.get_loaders(test_images, test_labels, test_images, test_labels)
        
        # Evaluate
        predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            model_idx,
            f'results/model_{model_idx}/confusion_matrix.jpg'
        )
        
        # Find and save example images
        find_example_images(predictions, true_labels, probabilities, test_images, model_idx)
        
        # Generate Grad-CAM visualizations
        generate_gradcam(model, model_idx, device)
        
        # Save metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        with open(f'results/model_{model_idx}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        all_metrics[f'model_{model_idx}'] = metrics
        logger.info(f"Metrics for model {model_idx} saved to results/model_{model_idx}/metrics.json")
        
        # Clear GPU memory after each model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save combined metrics
    with open('results/combined_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    logger.info("Validation completed. Results saved to results/combined_metrics.json")

if __name__ == '__main__':
    main()
