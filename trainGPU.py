import os
import logging
import argparse
from datetime import datetime

from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.fashion_dataset import FashionDataset
from models.fashion_cnn import FashionHybridModel, FashionMultiTaskLoss

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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        attributes = batch['attributes'].to(device)
        
        # Prepare targets
        targets = {
            'category_labels': batch['category_label'].to(device),
            'category_type_labels': batch['category_type'].to(device),
            'attribute_targets': attributes,
            'compatibility_targets': torch.ones(images.size(0), 1).to(device)  # Dummy compatibility targets
        }
        
        # Forward pass
        outputs = model(images, heatmaps, attributes)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k != 'total_loss':
                loss_components[k.replace('_loss', '')] += v
        
        if batch_idx % 10 == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}')
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_category = 0
    correct_type = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            attributes = batch['attributes'].to(device)
            
            targets = {
                'category_labels': batch['category_label'].to(device),
                'category_type_labels': batch['category_type'].to(device),
                'attribute_targets': attributes,
                'compatibility_targets': torch.ones(images.size(0), 1).to(device)
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
    
    avg_loss = total_loss / len(val_loader)
    category_acc = 100. * correct_category / total
    type_acc = 100. * correct_type / total
    
    return avg_loss, category_acc, type_acc

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
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
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = FashionHybridModel(
        num_categories=args.num_categories,
        num_category_types=args.num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    ).to(device)
    
    # Initialize criterion and optimizer
    criterion = FashionMultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, category_acc, type_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f'Epoch: {epoch}')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        logger.info(f'Category Accuracy: {category_acc:.2f}%')
        logger.info(f'Type Accuracy: {type_acc:.2f}%')
        
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/category', category_acc, epoch)
        writer.add_scalar('Accuracy/type', type_acc, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Update learning rate
        scheduler.step(val_loss)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion Model Training Script')
    
    # Data arguments
    parser.add_argument('--train-metadata', type=str, required=True, help='Path to training metadata CSV')
    parser.add_argument('--val-metadata', type=str, required=True, help='Path to validation metadata CSV')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--heatmaps-dir', type=str, required=True, help='Path to heatmaps directory')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50', help='CNN backbone architecture')
    parser.add_argument('--num-categories', type=int, default=50, help='Number of clothing categories')
    parser.add_argument('--num-category-types', type=int, default=4, help='Number of category types')
    parser.add_argument('--num-attributes', type=int, default=50, help='Number of attributes')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='runs', help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    # Manually define args (for debugging)
    args = parser.parse_args([
        '--train-metadata', 'X:\dissertation\StyleSyncProject\zsehran\metadata_updated_train.csv',
        '--val-metadata', 'X:\dissertation\StyleSyncProject\zsehran\metadata_updated_val.csv',
        '--images-dir', 'X:\dissertation\StyleSyncProject\data\processed\cropped_images',
        '--heatmaps-dir', 'X:\dissertation\StyleSyncProject\data\processed\heatmaps',
        '--batch-size', '16',
        '--epochs', '25',
        '--learning-rate', '0.0005',
        '--num-workers', '0'
    ])
    main(args)