import argparse
import logging
import math
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
from tqdm import tqdm
from models.fashion_cnn import FashionHybridModel, FashionMultiTaskLoss, compute_f1_multilabel, compute_f1_multiclass
from utils.fashion_dataset import FashionDataset
from torch.utils.data import Subset

warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger('TrainingScript')

# Print CUDA information
def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

  
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, total_epochs):
    model.train()
    total_loss = 0
    loss_components = {
        'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0,
        'image_aux': 0, 'heatmap_aux': 0, 'attribute_aux': 0, 'total_aux': 0
    }
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
    total_samples = 0
    num_batches = len(train_loader)  # Total number of batches per epoch

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, batch in progress_bar:
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)

        # Extract main category labels for mixup
        category_labels = batch['category_label'].to(device, non_blocking=True)
        # Apply mixup to images and category labels
        mixed_images, target_a, target_b, lam = mixup_data(images, category_labels, alpha=0.2)

        # Ensure batch is not empty
        if images.shape[0] == 0 or heatmaps.shape[0] == 0 or attributes.shape[0] == 0:
            raise RuntimeError(f"Empty batch received at index {batch_idx}")

        # Prepare other targets (for category type, attributes, compatibility)
        # Here we do not mixup these targets; they remain unchanged.
        targets = {
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes
        }
        if 'compatibility_targets' in batch:
            targets['compatibility_targets'] = batch['compatibility_targets'].to(device, non_blocking=True)
        else:
            targets['compatibility_targets'] = torch.zeros(images.size(0), 1).to(device, non_blocking=True)

        # Reset gradients
        optimizer.zero_grad()

        try:
            # Forward pass with mixed precision using updated autocast syntax and mixed images
            with torch.amp.autocast('cuda'):
                # Use mixed_images instead of original images
                outputs = model(mixed_images, heatmaps, attributes)
                # Compute category loss using mixup targets
                loss_cat_a = criterion.category_loss_fn(outputs['category_logits'], target_a)
                loss_cat_b = criterion.category_loss_fn(outputs['category_logits'], target_b)
                mixup_loss = lam * loss_cat_a + (1 - lam) * loss_cat_b

                # Compute other losses normally
                category_type_loss = criterion.category_type_loss_fn(
                    outputs['category_type_logits'], targets['category_type_labels'])
                attribute_loss = criterion.attribute_loss_fn(
                    outputs['attribute_preds'], targets['attribute_targets'])
                compatibility_loss = criterion.compatibility_loss_fn(
                    outputs['compatibility_score'], targets['compatibility_targets'])

                # Compute auxiliary losses (using target_a for category auxiliary branch)
                image_aux_loss = criterion.category_loss_fn(outputs['image_aux_logits'], target_a)
                heatmap_aux_loss = criterion.category_type_loss_fn(
                    outputs['heatmap_aux_logits'], targets['category_type_labels'])
                attribute_aux_loss = criterion.attribute_loss_fn(
                    outputs['attribute_aux_logits'], targets['attribute_targets'])
                total_aux_loss = image_aux_loss + heatmap_aux_loss + attribute_aux_loss

                # Combine losses with respective weights
                loss = (criterion.category_weight * mixup_loss +
                        criterion.category_type_weight * category_type_loss +
                        criterion.attribute_weight * attribute_loss +
                        criterion.compatibility_weight * compatibility_loss +
                        criterion.aux_weight * total_aux_loss)

                # Prepare loss dictionary for logging
                loss_dict = {
                    'category_loss': mixup_loss.item(),
                    'category_type_loss': category_type_loss.item(),
                    'attribute_loss': attribute_loss.item(),
                    'compatibility_loss': compatibility_loss.item(),
                    'image_aux_loss': image_aux_loss.item(),
                    'heatmap_aux_loss': heatmap_aux_loss.item(),
                    'attribute_aux_loss': attribute_aux_loss.item(),
                    'total_aux_loss': total_aux_loss.item()
                }

        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            raise

        progress_bar.set_postfix(loss=loss.item())

        # Backward pass and optimization steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics (for reporting, use the original outputs and un-mixed labels for category)
        # For metrics, you might choose to use target_a (or the average of target_a and target_b)
        f1_category = compute_f1_multiclass(outputs['category_probs'], target_a)
        accuracy_category = (outputs['category_logits'].argmax(dim=1) == target_a).float().mean()
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
            key = k.replace('_loss', '')
            if key in loss_components:
                loss_components[key] += v
            else:
                logger.warning(f"Warning: Key {key} not found in loss_components, skipping update.")

        # Print batch progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f" [Batch Progress] Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

    # Calculate averages and log results (unchanged from your current script)
    avg_loss = total_loss / num_batches
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    logger.info(f"Epoch {epoch} Completed | Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train F1 Scores - Category: {avg_f1_category:.4f}, Category Type: {avg_f1_category_type:.4f}, Attributes: {avg_f1_attributes:.4f}")
    print(f"Train Accuracy - Category: {avg_accuracy_category:.4f}, Category Type: {avg_accuracy_category_type:.4f}, Attributes: {avg_accuracy_attributes:.4f}")

    logger.info(f"Train Loss: {avg_loss:.4f}")
    logger.info(f"Train F1 Category: {avg_f1_category:.4f}")
    logger.info(f"Train F1 Category Type: {avg_f1_category_type:.4f}")
    logger.info(f"Train F1 Attributes: {avg_f1_attributes:.4f}")
    logger.info(f"Train Accuracy Category: {avg_accuracy_category:.4f}")
    logger.info(f"Train Accuracy Category Type: {avg_accuracy_category_type:.4f}")
    logger.info(f"Train Accuracy Attributes: {avg_accuracy_attributes:.4f}")
    logger.info(f"Loss Components: {avg_components}")

    return avg_loss, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components



def validate(model, val_loader, criterion, device, epoch):
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
            if 'compatibility_targets' in batch:
                targets['compatibility_targets'] = batch['compatibility_targets'].to(device, non_blocking=True)
            else:
                targets['compatibility_targets'] = torch.zeros(images.size(0), 1).to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, heatmaps, attributes)
                loss, loss_dict = criterion(outputs, targets, epoch)

            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            predicted_attributes = (outputs['attribute_probs'] > 0.5).float()

            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()
            correct_attributes += ((predicted_attributes == targets['attribute_targets']).all(dim=1).sum().item())

            total_loss += loss.item()

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

            for k in loss_components.keys():
                loss_components[k] += loss_dict.get(k, 0)

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

    avg_components = {k: v / total_samples for k, v in loss_components.items()}

    print(f" Validation - Loss: {avg_loss:.4f}")
    print(f"  Validation F1 Scores - Category: {avg_f1_category:.4f}, Type: {avg_f1_category_type:.4f}, Attributes: {avg_f1_attributes:.4f}")
    print(f" Validation Accuracy - Category: {avg_accuracy_category:.4f}, Type: {avg_accuracy_category_type:.4f}, Attributes: {avg_accuracy_attributes:.4f}")

    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation F1 Scores: Category={avg_f1_category:.4f}, Type={avg_f1_category_type:.4f}, Attributes={avg_f1_attributes:.4f}")
    logger.info(f"Validation Accuracy: Category={avg_accuracy_category:.4f}, Type={avg_accuracy_category_type:.4f}, Attributes={avg_accuracy_attributes:.4f}")

    return avg_loss, category_acc, type_acc, attributes_acc, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components


def main(args):
    # Print CUDA information
    print_cuda_info()

    for handler in logger.handlers:
        handler.flush()

    logger.info("Starting training script...")

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using CUDA Device: {torch.cuda.get_device_name(0)}")
        scaler = torch.amp.GradScaler()
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, running on CPU! Mixed Precision Disabled.")
        scaler = None

    # Define output_dir with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for saving checkpoints
    epoch_dir = os.path.join(output_dir, 'Epochs')
    model_dir = os.path.join(epoch_dir, 'Runs1')
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(output_dir)

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # New: Random crop and resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # Increased rotation range
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Slightly increased jitter
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = FashionDataset(
        metadata_path=args.train_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=False,
        cache_size=100,
        validate_files=True,
    )

    val_dataset = FashionDataset(
        metadata_path=args.val_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=val_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    # --- Reduce Validation Set Size ---
    val_subset_fraction = 0.2  # Use 20% of validation samples
    val_subset_size = int(val_subset_fraction * len(val_dataset))
    indices = list(range(len(val_dataset)))
    random.shuffle(indices)
    val_subset_indices = indices[:val_subset_size]
    val_dataset = Subset(val_dataset, val_subset_indices)
    print(f"Validation subset size: {len(val_dataset)}")


    assert train_dataset.metadata['category_label'].min() == 0, "Error: Category labels should start from 0"
    assert train_dataset.metadata['category_label'].max() == 45, "Error: Category label range should be 0 to 45"
    logger.info(" Category labels are correctly indexed from 0 to 45.")

    category_counts = train_dataset.metadata['category_label'].value_counts().to_dict()
    print("Class distribution:", category_counts)
    assert 45 in category_counts, "Error: Category 45 is missing from dataset!"
    num_samples = len(train_dataset)
    for i in range(46):
        if i not in category_counts:
            category_counts[i] = 1
    max_count = max(category_counts.values())
    class_weights = {cat: max_count / category_counts[cat] for cat in category_counts}
    class_weights[45] *= 10
    sample_weights = [min(class_weights[label], 10) for label in train_dataset.metadata['category_label']]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    num_categories = len(train_dataset.metadata['category_label'].unique())
    num_category_types = len(train_dataset.metadata['category_type'].unique())
    print(f"Number of unique categories in dataset: {num_categories}")
    print(f"Number of unique category types in dataset: {num_category_types}")

    torch.cuda.empty_cache()
    model = FashionHybridModel(
        num_categories=num_categories,
        num_category_types=num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    )

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    if hasattr(model, 'module'):
        model.module.freeze_backbone(freeze=False)
    else:
        model.freeze_backbone(freeze=False)

    criterion = FashionMultiTaskLoss(attribute_weight=0.7).to(device)

    if hasattr(model, 'module'):
        backbone_params = list(model.module.image_encoder.parameters())
    else:
        backbone_params = list(model.image_encoder.parameters())
    other_params = [p for name, p in model.named_parameters() if 'image_encoder' not in name]

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': args.learning_rate * 0.1},
        {'params': other_params, 'lr': args.learning_rate}
    ], weight_decay=1e-3)

    warmup_epochs = 3

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / warmup_epochs
        else:
            eta_min_ratio = 1e-6 / args.learning_rate
            progress = (current_epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return eta_min_ratio + (1 - eta_min_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler()


    checkpoint_path = '/content/drive/MyDrive/Runs/12epochs5run/Epochs/epoch_1.pth'
    start_epoch = 11  # Default starting epoch
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")
        best_val_loss = float('inf')

    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        if epoch == 5:
            if hasattr(model, 'module'):
                model.module.freeze_backbone(False)
            else:
                model.freeze_backbone(False)

        train_loss, train_f1_category, train_f1_category_type, train_f1_attributes, \
        train_accuracy_category, train_accuracy_category_type, train_accuracy_attributes, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, args.epochs
        )

        val_loss, val_category_acc, val_type_acc, val_attributes_acc, val_f1_category, val_f1_category_type, val_f1_attributes, \
        val_accuracy_category, val_accuracy_category_type, val_accuracy_attributes, val_components = validate(
            model, val_loader, criterion, device, epoch
        )

        print(f" Validation Epoch {epoch} | Loss: {val_loss:.4f}")
        print(f"  Validation F1 Scores - Category: {val_f1_category:.4f}, Type: {val_f1_category_type:.4f}, Attributes: {val_f1_attributes:.4f}")
        print(f" Validation Accuracy - Category: {val_accuracy_category:.4f}, Type: {val_accuracy_category_type:.4f}, Attributes: {val_accuracy_attributes:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train_category', train_f1_category, epoch)
        writer.add_scalar('F1/val_category', val_f1_category, epoch)
        writer.add_scalar('F1/train_category_type', train_f1_category_type, epoch)
        writer.add_scalar('F1/val_category_type', val_f1_category_type, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(epoch_dir, f'epoch_{epoch}.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(model_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    writer.close()

# Define the args object
class Args:
    def __init__(self):
        self.train_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_train.csv'
        self.val_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_val.csv'
        self.images_dir = '/content/cropped_images1'
        self.heatmaps_dir = '/content/heatmaps1'
        self.backbone = 'resnet50'
        self.num_categories = 46
        self.num_category_types = 3
        self.num_attributes = 50
        self.batch_size = 512
        self.epochs = 20
        self.learning_rate =  0.0005
        self.num_workers = 1

args = Args()

if __name__ == '__main__':
    main(args)