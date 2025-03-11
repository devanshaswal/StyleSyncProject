#TRAINING_SCRIPT
import argparse
import logging
import os
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.fashion_cnn import FashionHybridModel, FashionMultiTaskLoss, compute_f1_multilabel
from utils.fashion_dataset import FashionDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

        # Debugging: Print input shapes
        logger.info(f"Batch {batch_idx} - Input shapes:")
        logger.info(f"Images: {images.shape}, Heatmaps: {heatmaps.shape}, Attributes: {attributes.shape}")

        # Check for NaN or Inf values in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.error("Images contain NaN or Inf values!")
            continue
        if torch.isnan(heatmaps).any() or torch.isinf(heatmaps).any():
            logger.error("Heatmaps contain NaN or Inf values!")
            continue
        if torch.isnan(attributes).any() or torch.isinf(attributes).any():
            logger.error("Attributes contain NaN or Inf values!")
            continue
        if torch.isnan(targets['category_labels']).any() or torch.isinf(targets['category_labels']).any():
            logger.error("Category labels contain NaN or Inf values!")
            continue
        if torch.isnan(targets['category_type_labels']).any() or torch.isinf(targets['category_type_labels']).any():
            logger.error("Category type labels contain NaN or Inf values!")
            continue

        # Check label values
        if (targets['category_labels'] >= args.num_categories).any():
            logger.error(f"Invalid category labels found: {targets['category_labels']}")
            continue
        if (targets['category_type_labels'] >= args.num_category_types).any():
            logger.error(f"Invalid category type labels found: {targets['category_type_labels']}")
            continue

        # Forward pass with mixed precision
        optimizer.zero_grad()
        try:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images, heatmaps, attributes)
        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            logger.error(f"Input shapes - Images: {images.shape}, Heatmaps: {heatmaps.shape}, Attributes: {attributes.shape}")
            continue  # Skip this batch and move to the next one

        # Check model output shapes
        assert outputs['category_logits'].shape[1] == args.num_categories, \
            f"Expected category_logits shape [batch_size, {args.num_categories}], got {outputs['category_logits'].shape}"
        assert outputs['category_type_logits'].shape[1] == args.num_category_types, \
            f"Expected category_type_logits shape [batch_size, {args.num_category_types}], got {outputs['category_type_logits'].shape}"

        # Check for NaN or Inf values in model outputs
        for key, value in outputs.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                logger.warning(f"⚠️ Warning: {key} contains NaN or Inf values!")

        # Compute loss
        loss, loss_dict = criterion(outputs, targets)

        # Compute metrics
        f1_category = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
        accuracy_category = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

        f1_category_type = compute_f1_multilabel(outputs['category_type_probs'], targets['category_type_labels'])
        accuracy_category_type = (outputs['category_type_logits'].argmax(dim=1) == targets['category_type_labels']).float().mean()

        f1_attributes = compute_f1_multilabel(outputs['attribute_probs'], targets['attribute_targets'])
        accuracy_attributes = (outputs['attribute_probs'] > 0.5).float().mean()  # If thresholding

        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        scaler.step(optimizer)
        scaler.update()

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

    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

    # Print all metrics for transparency
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
                'attribute_targets': attributes,
                'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
            }

            # Debugging: Print shapes and min/max values before loss computation
            print("\n==== DEBUG: VALIDATION BATCH ====")
            print(f"Category Labels Shape: {targets['category_labels'].shape}, Min: {targets['category_labels'].min()}, Max: {targets['category_labels'].max()}")
            print(f"Category Type Labels Shape: {targets['category_type_labels'].shape}, Min: {targets['category_type_labels'].min()}, Max: {targets['category_type_labels'].max()}")
            print(f"Attribute Targets Shape: {targets['attribute_targets'].shape}, Min: {targets['attribute_targets'].min()}, Max: {targets['attribute_targets'].max()}")
            print(f"Compatibility Targets Shape: {targets['compatibility_targets'].shape}, Min: {targets['compatibility_targets'].min()}, Max: {targets['compatibility_targets'].max()}")

            # Ensure shapes match before loss computation
            assert outputs['category_logits'].shape[0] == targets['category_labels'].shape[0], "Mismatch in category logits and labels"
            assert outputs['category_type_logits'].shape[0] == targets['category_type_labels'].shape[0], "Mismatch in category type logits and labels"
            assert outputs['attribute_preds'].shape == targets['attribute_targets'].shape, \
                   f"Mismatch in attribute predictions {outputs['attribute_preds'].shape} and targets {targets['attribute_targets'].shape}"

            # Check for NaN or Inf values in model outputs
            for key, value in outputs.items():
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"⚠️ Warning: {key} contains NaN or Inf values!")


            # Mixed precision validation
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images, heatmaps, attributes)
                loss, loss_dict = criterion(outputs, targets)

            # Compute individual loss components and accumulate them
            loss_components['category'] += loss_dict.get('category', 0)
            loss_components['category_type'] += loss_dict.get('category_type', 0)
            loss_components['attribute'] += loss_dict.get('attribute', 0)
            loss_components['compatibility'] += loss_dict.get('compatibility', 0)

            # Compute metrics
            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            predicted_attributes = (outputs['attribute_probs'] > 0.5).float()

            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()
            correct_attributes += ((predicted_attributes == targets['attribute_targets']).all(dim=1).sum().item())

            total_loss += loss.item()

            # Compute F1 score and accuracy
            f1_category = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
            accuracy_category = (predicted_category == targets['category_labels']).float().mean()

            f1_category_type = compute_f1_multilabel(outputs['category_type_probs'], targets['category_type_labels'])
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

    # Logging
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation Loss Components: {avg_components}")

     # Print validation metrics
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Category Accuracy: {category_acc:.2f}%")
    print(f"Type Accuracy: {type_acc:.2f}%")
    print(f"Attributes Accuracy: {attributes_acc:.2f}%")
    print(f"Validation F1 Category: {avg_f1_category:.4f}")
    print(f"Validation F1 Category Type: {avg_f1_category_type:.4f}")
    print(f"Validation F1 Attributes: {avg_f1_attributes:.4f}")
    print(f"Validation Accuracy Category: {avg_accuracy_category:.4f}")
    print(f"Validation Accuracy Category Type: {avg_accuracy_category_type:.4f}")
    print(f"Validation Accuracy Attributes: {avg_accuracy_attributes:.4f}")
    print(f"Validation Loss Components: {avg_components}")

    return avg_loss, category_acc, type_acc, attributes_acc, avg_f1_category, avg_f1_category_type, avg_f1_attributes, avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components

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
        shuffle=False,
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
    # model.freeze_backbone(freeze=True)
    # Freeze the backbone initially
    if hasattr(model, 'module'):  # If using DataParallel
        model.module.freeze_backbone(freeze=True)
    else:
        model.freeze_backbone(freeze=True)

    # Initialize criterion and optimizer
    criterion = FashionMultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    # Cosine Annealing LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    epochs_without_improvement = 0

    # If you've passed a certain number of epochs, unfreeze the backbone
    for epoch in range(args.epochs):
        # Unfreeze backbone after epoch 5
        if epoch == 5:
            if hasattr(model, 'module'):
                model.module.freeze_backbone(False)
            else:
                model.freeze_backbone(False)

        # Train
        train_loss, avg_f1_category, avg_f1_category_type, avg_f1_attributes, \
        avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components = train_epoch( model, train_loader, criterion, optimizer, device, epoch, scaler )

        # Validate
        val_loss, category_acc, type_acc, attributes_acc, avg_f1_category, avg_f1_category_type, avg_f1_attributes, \
        avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components = validate(model, val_loader, criterion, device)



        # Print basic information
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')

        # Print key performance metrics
        print(f'Train F1 Category: {avg_f1_category:.4f}')
        print(f'Train F1 Category Type: {avg_f1_category_type:.4f}')
        print(f'Train F1 Attributes: {avg_f1_attributes:.4f}')
        print(f'Train Accuracy Category: {avg_accuracy_category:.4f}')
        print(f'Train Accuracy Category Type: {avg_accuracy_category_type:.4f}')
        print(f'Train Accuracy Attributes: {avg_accuracy_attributes:.4f}')

        print(f'Val F1 Category: {avg_f1_category:.4f}')
        print(f'Val F1 Category Type: {avg_f1_category_type:.4f}')
        print(f'Val F1 Attributes: {avg_f1_attributes:.4f}')
        print(f'Val Accuracy Category: {avg_accuracy_category:.4f}')
        print(f'Val Accuracy Category Type: {avg_accuracy_category_type:.4f}')
        print(f'Val Accuracy Attributes: {avg_accuracy_attributes:.4f}')

        # Optionally, print validation loss components for debugging (if it's important)
        print(f'Validation Loss Components: {avg_components}')
        print(f'Train Loss Components: {avg_components}')


        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Accuracy logs
        writer.add_scalar('Accuracy/train', avg_accuracy_category, epoch)  # Can choose category/overall accuracy
        writer.add_scalar('Accuracy/val', avg_accuracy_category, epoch)

        # F1 score logs
        writer.add_scalar('F1/train', avg_f1_category, epoch)  # Choose category F1 for simplicity
        writer.add_scalar('F1/val', avg_f1_category, epoch)


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
        # scheduler.step(val_loss)
        # Step the scheduler after every epoch
        scheduler.step()
        # Inside the training loop, after scheduler.step()
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
        self.epochs = 2
        self.learning_rate = 0.001
        self.num_workers = 2

# Create the args object
args = Args()

# Run the script
if __name__ == '__main__':
    main(args)
