







import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.fashion_cnn import (FashionHybridModel, FashionMultiTaskLoss,
                                compute_f1_multilabel)
from utils.fashion_dataset import FashionDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TrainingScript')

# Print CUDA information
def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    total_f1 = 0
    total_accuracy = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)

        # Prepare targets
        targets = {
            'category_labels': batch['category_label'].to(device, non_blocking=True),
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes,
            'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
        }

        # Forward pass with mixed precision
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images, heatmaps, attributes)
            loss, loss_dict = criterion(outputs, targets)
            
        # Compute F1 score
        f1 = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
        accuracy = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        total_f1 += f1
        total_accuracy += accuracy
        total_samples += 1
        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_components[k.replace('_loss', '')] += v

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')

    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_f1 = total_f1 / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

    return avg_loss, avg_f1, avg_accuracy, avg_components

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_category = 0
    correct_type = 0
    total = 0
    total_f1 = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            heatmaps = batch['heatmap'].to(device, non_blocking=True)
            attributes = batch['attributes'].to(device, non_blocking=True)

            targets = {
                'category_labels': batch['category_label'].to(device, non_blocking=True),
                'category_type_labels': batch['category_type'].to(device, non_blocking=True),
                'attribute_targets': attributes,
                'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
            }

            outputs = model(images, heatmaps, attributes)
            loss, _ = criterion(outputs, targets)

            # Calculate accuracy
            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()

            total_loss += loss.item()

            # Compute F1 score and accuracy
            f1 = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
            accuracy = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

            total_f1 += f1
            total_accuracy += accuracy
            total_samples += 1

    avg_loss = total_loss / len(val_loader)
    category_acc = 100. * correct_category / total
    type_acc = 100. * correct_type / total
    avg_f1 = total_f1 / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, category_acc, type_acc, avg_f1, avg_accuracy

def main(args):
    # Print CUDA information
    print_cuda_info()

    # empty the cache
    torch.cuda.empty_cache()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory in Google Drive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(output_dir)

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = FashionDataset(
        metadata_path=args.train_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    val_dataset = FashionDataset(
        metadata_path=args.val_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model
    model = FashionHybridModel(
        num_categories=args.num_categories,
        num_category_types=args.num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Initialize criterion and optimizer
    criterion = FashionMultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_f1, train_accuracy, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )

        # Validate
        val_loss, category_acc, type_acc, val_f1, val_accuracy = validate(model, val_loader, criterion, device)

        # Log metrics
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}')
        print(f'Category Accuracy: {category_acc:.2f}%')
        print(f'Type Accuracy: {type_acc:.2f}%')

        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        # Save model for each epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join("/content/drive/MyDrive/Runs/Epochs", f'epoch_{timestamp}_{epoch}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Update learning rate
        scheduler.step(val_loss)

    writer.close()

# Define the args object
class Args:
    def __init__(self):
        self.train_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_train.csv'
        self.val_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_val.csv'
        self.images_dir = '/content/cropped_images1'
        self.heatmaps_dir = '/content/heatmaps1'
        self.backbone = 'resnet50'
        self.num_categories = 50
        self.num_category_types = 4
        self.num_attributes = 50
        self.batch_size = 512
        self.epochs = 2
        self.learning_rate = 0.001
        self.num_workers = 2

# Create the args object
args = Args()

# Run the script
if __name__ == '__main__':
    main(args)




#TRAINING_SCRIPT

import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error

from models.fashion_cnn import FashionHybridModel, FashionMultiTaskLoss, compute_f1_multilabel
from utils.fashion_dataset import FashionDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TrainingScript')

# Print CUDA information
def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    total_f1 = 0
    total_accuracy = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)

        # Prepare targets
        targets = {
            'category_labels': batch['category_label'].to(device, non_blocking=True),
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes,
            'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
        }

        # Forward pass with mixed precision
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images, heatmaps, attributes)
            loss, loss_dict = criterion(outputs, targets)
            
        # Compute F1 score
        f1 = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
        accuracy = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        total_f1 += f1
        total_accuracy += accuracy
        total_samples += 1
        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_components[k.replace('_loss', '')] += v

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')

    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_f1 = total_f1 / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

    return avg_loss, avg_f1, avg_accuracy, avg_components

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_category = 0
    correct_type = 0
    total = 0
    total_f1 = 0
    total_accuracy = 0
    total_samples = 0


    # Loss components for logging
    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            heatmaps = batch['heatmap'].to(device, non_blocking=True)
            attributes = batch['attributes'].to(device, non_blocking=True)

            targets = {
                'category_labels': batch['category_label'].to(device, non_blocking=True),
                'category_type_labels': batch['category_type'].to(device, non_blocking=True),
                'attribute_targets': attributes,
                'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
            }

            outputs = model(images, heatmaps, attributes)
            loss, _ = criterion(outputs, targets)

            # Calculate accuracy
            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()

            total_loss += loss.item()

            # Compute F1 score and accuracy
            attribute_preds = (outputs['attribute_probs'] > 0.5).float()  # Threshold at 0.5 for binary predictions
            f1 = compute_f1_multilabel(attribute_preds, targets['attribute_targets'])
            accuracy = (attribute_preds == targets['attribute_targets']).float().mean()

            total_f1 += f1
            total_accuracy += accuracy
            total_samples += 1

    avg_loss = total_loss / len(val_loader)
    category_acc = 100. * correct_category / total
    type_acc = 100. * correct_type / total
    avg_f1 = total_f1 / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, category_acc, type_acc, avg_f1, avg_accuracy

def main(args):
    # Print CUDA information
    print_cuda_info()

    # empty the cache
    torch.cuda.empty_cache()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory in Google Drive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(output_dir)

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = FashionDataset(
        metadata_path=args.train_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    val_dataset = FashionDataset(
        metadata_path=args.val_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model
    model = FashionHybridModel(
        num_categories=args.num_categories,
        num_category_types=args.num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Freeze the backbone initially
    model.freeze_backbone(freeze=True)

    # Initialize criterion and optimizer
    criterion = FashionMultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # If you've passed a certain number of epochs, unfreeze the backbone
        if epoch >= 5:  # Change this to any number of epochs you prefer
            model.freeze_backbone(freeze=False)
      
        # Train
        train_loss, train_f1, train_accuracy, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )

        # Validate
        val_loss, category_acc, type_acc, val_f1, val_accuracy = validate(model, val_loader, criterion, device)

        # Log metrics
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}')
        print(f'Category Accuracy: {category_acc:.2f}%')
        print(f'Type Accuracy: {type_acc:.2f}%')

        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        # Save model for each epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join("/content/drive/MyDrive/Runs/Epochs", f'epoch_{timestamp}_{epoch}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Update learning rate
        scheduler.step(val_loss)

    writer.close()

# Define the args object
class Args:
    def __init__(self):
        self.train_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_train.csv'
        self.val_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_val.csv'
        self.images_dir = '/content/cropped_images1'
        self.heatmaps_dir = '/content/heatmaps1'
        self.backbone = 'resnet50'
        self.num_categories = 50
        self.num_category_types = 4
        self.num_attributes = 50
        self.batch_size = 512
        self.epochs = 2
        self.learning_rate = 0.001
        self.num_workers = 2

# Create the args object
args = Args()

# Run the script
if __name__ == '__main__':
    main(args)
