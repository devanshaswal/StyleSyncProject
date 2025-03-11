import argparse
import logging
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler  # Correct import for mixed precision
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")


# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.modelerror import (FashionHybridModel, FashionMultiTaskLoss,
                               compute_f1_multiclass, compute_f1_multilabel)
from utils.data_collab import FashionDataset

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
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
    total_samples = 0
    num_batches = len(train_loader)  # Total number of batches per epoch

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)

        # Ensure batch is not empty
        if images.shape[0] == 0 or heatmaps.shape[0] == 0 or attributes.shape[0] == 0:
            raise RuntimeError(f"Empty batch received at index {batch_idx}")

        # Prepare targets
        targets = {
            'category_labels': batch['category_label'].to(device, non_blocking=True),
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes
        }

        # Check if batch contains compatibility_targets, otherwise set default value
        if 'compatibility_targets' in batch:
            targets['compatibility_targets'] = batch['compatibility_targets'].to(device, non_blocking=True)
        else:
            targets['compatibility_targets'] = torch.zeros(images.size(0), 1).to(device, non_blocking=True)

        # Reset gradients
        optimizer.zero_grad()

        try:
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, heatmaps, attributes)

            # Compute loss
            loss, loss_dict = criterion(outputs, targets)

        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            raise

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        # print(f" Before optimizer step - Loss: {loss.item():.4f}")
        scaler.step(optimizer)
        scaler.update()
        # print(f" After optimizer step - Moving to next batch...")

        # Check for NaN gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f" NaN detected in gradients of {name}, stopping training!")
        #         raise RuntimeError(f"NaN gradients found in {name}.")

        
        # Compute metrics
        f1_category = compute_f1_multiclass(outputs['category_probs'], targets['category_labels'])
        accuracy_category = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()
        f1_category_type = compute_f1_multiclass(outputs['category_type_probs'], targets['category_type_labels'])
        accuracy_category_type = (outputs['category_type_logits'].argmax(dim=1) == targets['category_type_labels']).float().mean()
        f1_attributes = compute_f1_multilabel(outputs['attribute_probs'], targets['attribute_targets'])
        accuracy_attributes = (outputs['attribute_probs'] > 0.5).float().mean()

        # Update metrics
        total_loss += loss.item()
        total_f1_category += f1_category
        total_f1_category_type += f1_category_type
        total_f1_attributes += f1_attributes
        total_accuracy_category += accuracy_category
        total_accuracy_category_type += accuracy_category_type
        total_accuracy_attributes += accuracy_attributes
        total_samples += 1

        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_components[k.replace('_loss', '')] += v

        # Print batch progress every 10 batches (instead of full category lists)
        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    # Print learning rate after each epoch
    # Print training stats after each epoch
    print(f"Epoch {epoch} Completed | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train F1 Scores - Category: {avg_f1_category:.4f}, Category Type: {avg_f1_category_type:.4f}, Attributes: {avg_f1_attributes:.4f}")
    print(f"Train Accuracy - Category: {avg_accuracy_category:.4f}, Category Type: {avg_accuracy_category_type:.4f}, Attributes: {avg_accuracy_attributes:.4f}")


    # Log metrics
    logger.info(f"Train Loss: {avg_loss:.4f}")
    logger.info(f"Train F1 Category: {avg_f1_category:.4f}")
    logger.info(f"Train F1 Category Type: {avg_f1_category_type:.4f}")
    logger.info(f"Train F1 Attributes: {avg_f1_attributes:.4f}")
    logger.info(f"Train Accuracy Category: {avg_accuracy_category:.4f}")
    logger.info(f"Train Accuracy Category Type: {avg_accuracy_category_type:.4f}")
    logger.info(f"Train Accuracy Attributes: {avg_accuracy_attributes:.4f}")
    logger.info(f"Loss Components: {avg_components}")

    return avg_loss, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_category = 0
    correct_type = 0
    correct_attributes = 0
    total = 0
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
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
                'attribute_targets': attributes
            }

            # Mixed precision validation
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, heatmaps, attributes)
                loss, loss_dict = criterion(outputs, targets)

            # Compute predictions
            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            predicted_attributes = (outputs['attribute_probs'] > 0.5).float()

            # Compute correct predictions
            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()
            correct_attributes += ((predicted_attributes == targets['attribute_targets']).all(dim=1).sum().item())

            total_loss += loss.item()

            # Compute F1 scores & accuracy
            f1_category = compute_f1_multiclass(outputs['category_probs'], targets['category_labels'])
            accuracy_category = (predicted_category == targets['category_labels']).float().mean()

            f1_category_type = compute_f1_multiclass(outputs['category_type_probs'], targets['category_type_labels'])
            accuracy_category_type = (predicted_type == targets['category_type_labels']).float().mean()

            f1_attributes = compute_f1_multilabel(predicted_attributes, targets['attribute_targets'])
            accuracy_attributes = (predicted_attributes == targets['attribute_targets']).float().mean()

            total_f1_category += f1_category
            total_f1_category_type += f1_category_type
            total_f1_attributes += f1_attributes
            total_accuracy_category += accuracy_category
            total_accuracy_category_type += accuracy_category_type
            total_accuracy_attributes += accuracy_attributes
            total_samples += 1

            # Accumulate individual loss components
            for k in loss_components.keys():
                loss_components[k] += loss_dict.get(k, 0)

    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    category_acc = 100. * correct_category / total
    type_acc = 100. * correct_type / total
    attributes_acc = 100. * correct_attributes / total
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples

    # Calculate average of loss components
    avg_components = {k: v / total_samples for k, v in loss_components.items()}

    # Print Validation Metrics
    print(f" Validation - Loss: {avg_loss:.4f}")
    print(f"  Validation F1 Scores - Category: {avg_f1_category:.4f}, Type: {avg_f1_category_type:.4f}, Attributes: {avg_f1_attributes:.4f}")
    print(f" Validation Accuracy - Category: {avg_accuracy_category:.4f}, Type: {avg_accuracy_category_type:.4f}, Attributes: {avg_accuracy_attributes:.4f}")

    # Logging
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation F1 Scores: Category={avg_f1_category:.4f}, Type={avg_f1_category_type:.4f}, Attributes={avg_f1_attributes:.4f}")
    logger.info(f"Validation Accuracy: Category={avg_accuracy_category:.4f}, Type={avg_accuracy_category_type:.4f}, Attributes={avg_accuracy_attributes:.4f}")

    return avg_loss, category_acc, type_acc, attributes_acc, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components


def main(args):
    # Print CUDA information
    print_cuda_info()

    # Empty the cache
    torch.cuda.empty_cache()

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(" Warning: CUDA not available, running on CPU!")


    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(output_dir)

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    # Debugging: Ensure category labels are correctly mapped
    assert train_dataset.metadata['category_label'].min() == 0, "Error: Category labels should start from 0"
    assert train_dataset.metadata['category_label'].max() == 45, "Error: Category label range should be 0 to 45"
    logger.info(" Category labels are correctly indexed from 0 to 45.")


    # Compute class frequencies
    category_counts = train_dataset.metadata['category_label'].value_counts().to_dict()
    print("Class distribution:", category_counts)
    assert 45 in category_counts, "Error: Category 45 is missing from dataset!"
    num_samples = len(train_dataset)

    # Ensure all categories exist, including 45
    for i in range(46):
        if i not in category_counts:
            category_counts[i] = 1  # Assign a small count to prevent division by zero

    # Compute weights: Use the highest count divided by each category count
    max_count = max(category_counts.values())
    class_weights = {cat: max_count / category_counts[cat] for cat in category_counts}

    #  Boost category 45 even more
    class_weights[45] *= 10  # Increase weight for category 45

    # Apply new weights
    # Prevent extremely high sample weights for category 45
    sample_weights = [min(class_weights[label], 10) for label in train_dataset.metadata['category_label']]


    # Create a sampler that forces balance
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

    # def force_category_45(dataset, batch_size):
    #     """Ensure every batch contains at least one sample of category 45."""
    #     dataset_45 = [sample for sample in dataset if sample['category_label'].item() == 45]

    #     while True:
    #         batch = random.sample(dataset, batch_size - 1)  # Pick batch_size - 1 samples
    #         batch.append(random.choice(dataset_45))  # Always add one category 45 sample
    #         yield batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,   #force_category_45(train_dataset, args.batch_size), 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Get the correct number of valid categories
    num_categories = len(train_dataset.metadata['category_label'].unique())
    num_category_types = len(train_dataset.metadata['category_type'].unique())
    print(f"Number of unique categories in dataset: {num_categories}")
    print(f"Number of unique category types in dataset: {num_category_types}")

    # Initialize model
    model = FashionHybridModel(
        num_categories=num_categories,
        num_category_types=num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    )

    # Disable BatchNorm running statistics to prevent training hangs
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False # Prevent NaNs from BatchNorm


    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Freeze the backbone initially
    # Unfreeze the backbone so it updates during training
    if hasattr(model, 'module'):
        model.module.freeze_backbone(freeze=False)
    else:
        model.freeze_backbone(freeze=False)


    # Initialize criterion and optimizer
    criterion = FashionMultiTaskLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # Increased LR

    # Cosine Annealing LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision training
    scaler = GradScaler()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Unfreeze backbone after epoch 5
        if epoch == 5:
            if hasattr(model, 'module'):
                model.module.freeze_backbone(False)
            else:
                model.freeze_backbone(False)

        # Train
        train_loss, train_f1_category, train_f1_category_type, train_f1_attributes, \
        train_accuracy_category, train_accuracy_category_type, train_accuracy_attributes, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )

        # Validate
        val_loss, val_category_acc, val_type_acc, val_attributes_acc, val_f1_category, val_f1_category_type, val_f1_attributes, \
        val_accuracy_category, val_accuracy_category_type, val_accuracy_attributes, val_components = validate(
            model, val_loader, criterion, device
        )

        # Print Validation Metrics
        print(f" Validation Epoch {epoch} | Loss: {val_loss:.4f}")
        print(f"  Validation F1 Scores - Category: {val_f1_category:.4f}, Type: {val_f1_category_type:.4f}, Attributes: {val_f1_attributes:.4f}")
        print(f" Validation Accuracy - Category: {val_accuracy_category:.4f}, Type: {val_accuracy_category_type:.4f}, Attributes: {val_accuracy_attributes:.4f}")


        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train_category', train_f1_category, epoch)
        writer.add_scalar('F1/val_category', val_f1_category, epoch)
        writer.add_scalar('F1/train_category_type', train_f1_category_type, epoch)
        writer.add_scalar('F1/val_category_type', val_f1_category_type, epoch)

        # Save model for each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(output_dir, f'epoch_{epoch}.pth'))

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
        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    writer.close()

# Define the args object
class Args:
    def __init__(self):
        self.train_metadata = 'X:\dissertation\StyleSyncProject\zsehran\metadata_updated_train.csv'
        self.val_metadata = 'X:\dissertation\StyleSyncProject\zsehran\metadata_updated_val.csv'
        self.images_dir = "data/processed/cropped_images1"
        self.heatmaps_dir = "data/processed/heatmaps1"
        self.backbone = 'resnet50'
        self.num_categories = None
        self.num_category_types = 3
        self.num_attributes = 50
        self.batch_size = 32
        self.epochs = 2
        self.learning_rate = 0.001
        self.num_workers = 2

# Create the args object
args = Args()

# Run the script
if __name__ == '__main__':
    main(args)


