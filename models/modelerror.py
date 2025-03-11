from typing import Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FashionHybridModel')



def compute_f1_multiclass(preds, targets):
    """Computes F1-score for multi-class classification."""
    preds = preds.argmax(dim=1).cpu().numpy()  # Convert logits to class indices
    targets = targets.cpu().numpy()  # Convert targets to numpy
    return f1_score(targets, preds, average='macro', zero_division=1)  # Compute F1-score


def compute_f1_multilabel(preds, targets):
    """Computes F1-score for multi-label classification."""
    preds = (preds > 0.5).int().cpu().numpy()  # Convert logits to binary labels (threshold at 0.5)
    targets = targets.cpu().numpy()  # Convert ground truth to numpy
    return f1_score(targets, preds, average='macro', zero_division=1)  # Compute F1-score


class FashionHybridModel(nn.Module):
    """
    Hybrid CNN model for fashion recommendation.
    Processes:
    - Cropped images (224x224 RGB)
    - Landmark heatmaps (224x224 single-channel)
    - Attribute metadata (50 attributes)
    Outputs:
    - Category predictions (multi-class)
    - Category type predictions (multi-class)
    - Compatibility scores (for outfit recommendations)
    """

    def __init__(self, num_categories: int = 46, num_category_types: int = 3, num_attributes: int = 50,
                 embed_dim: int = 256, use_pretrained: bool = True, dropout_rate: float = 0.5,
                 backbone: str = 'resnet50'):
        """
        Args:
            num_categories (int): Number of clothing categories (default: 46).
            num_category_types (int): Number of category types (default: 3).
            num_attributes (int): Number of attributes (default: 50).
            embed_dim (int): Dimension of the fused feature embedding (default: 256).
            use_pretrained (bool): Whether to use pretrained weights for the backbone.
            dropout_rate (float): Dropout rate for regularization (default: 0.5).
            backbone (str): CNN backbone architecture ('resnet50', 'resnet101', 'efficientnet_b3').
        """
        super().__init__()
        logger.info(f"Initializing FashionHybridModel with {backbone} backbone")
        logger.info(f"Parameters: num_categories={num_categories}, num_category_types={num_category_types}, "
                   f"num_attributes={num_attributes}, embed_dim={embed_dim}")

        # Initialize backbone
        self.backbone = backbone
        self.image_feature_dim = self._init_image_encoder(use_pretrained)

        # Heatmap encoder (custom CNN)
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 1 channel (heatmap)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 112, 112]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 56, 56]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [128, 28, 28]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # [256, 1, 1]
        )
        self.heatmap_feature_dim = 256  # Heatmap encoder output dimension
        logger.info(f"Heatmap encoder initialized with output dimension: {self.heatmap_feature_dim}")

        # Attribute encoder (fully connected network)
        self.attribute_encoder = nn.Sequential(
            nn.Linear(num_attributes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.attribute_feature_dim = 64  # Attribute encoder output dimension
        logger.info(f"Attribute encoder initialized with output dimension: {self.attribute_feature_dim}")

        # Total feature dimension after concatenation
        total_feature_dim = self.image_feature_dim + self.heatmap_feature_dim + self.attribute_feature_dim
        logger.info(f"Total feature dimension after concatenation: {total_feature_dim}")

        # Fusion layer (combine image, heatmap, and attribute features)
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classification heads
        # 1. Category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_categories)
        )

        # 2. Category type classifier
        self.category_type_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_category_types)
        )

        # 3. Attribute predictor (multi-label classification)
        self.attribute_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_attributes),
            # nn.Sigmoid()  # Multiple binary classifications
        )

        # 4. Compatibility head (predict outfit compatibility scores)
        self.compatibility_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output compatibility score between 0 and 1
        )

        logger.info("Model initialization complete")

    def _init_image_encoder(self, use_pretrained: bool) -> int:
        """Initialize the image encoder backbone based on selection"""
        if self.backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 2048
        elif self.backbone == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 2048
        elif self.backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 1536
        else:
            logger.warning(f"Unknown backbone {self.backbone}, falling back to resnet50")
            model = models.resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 2048

        # Remove the final fully connected layer
        if 'resnet' in self.backbone:
            self.image_encoder = nn.Sequential(*list(model.children())[:-1])
        else:  # EfficientNet
            self.image_encoder = model.features

        logger.info(f"Initialized {self.backbone} backbone with feature dimension: {feature_dim}")
        return feature_dim

    def forward(self, image: torch.Tensor, heatmap: torch.Tensor, attributes: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            image (torch.Tensor): Cropped image tensor of shape [batch_size, 3, 224, 224].
            heatmap (torch.Tensor): Heatmap tensor of shape [batch_size, 1, 224, 224].
            attributes (torch.Tensor): Attribute tensor of shape [batch_size, num_attributes].
            return_embeddings (bool): Whether to return intermediate embeddings (useful for visualization).

        Returns:
            dict: Dictionary containing model outputs:
                - category_logits: Category predictions [batch_size, num_categories].
                - category_type_logits: Category type predictions [batch_size, num_category_types].
                - attribute_preds: Attribute predictions [batch_size, num_attributes].
                - compatibility_score: Compatibility scores [batch_size, 1].
                - embeddings: Feature embeddings if return_embeddings=True [batch_size, embed_dim].
        """
        try:
            # Print input shapes only for the first batch of each epoch
            if not hasattr(self, 'input_logged'):
                logger.info(f"Input shapes - Image: {image.shape}, Heatmap: {heatmap.shape}, Attributes: {attributes.shape}")
                self.input_logged = True  # Prevent repeated logging


            # Image features
            img_feat = self.image_encoder(image)  # [batch_size, channels, 1, 1] or [batch_size, channels, H, W]
            img_feat = torch.flatten(img_feat, 1)  # [batch_size, image_feature_dim]

            # Heatmap features
            heatmap_feat = self.heatmap_encoder(heatmap)  # [batch_size, heatmap_feature_dim, 1, 1]
            heatmap_feat = heatmap_feat.view(image.size(0), -1)  # [batch_size, heatmap_feature_dim]

            # Attribute features
            attr_feat = self.attribute_encoder(attributes)  # [batch_size, attribute_feature_dim]

            # Concatenate features
            fused_feat = torch.cat([img_feat, heatmap_feat, attr_feat], dim=1)
            fused_feat = self.fusion(fused_feat)  # [batch_size, embed_dim]
       
            # Print feature shapes only once per epoch
            if not hasattr(self, 'feature_shapes_logged'):
                logger.info(f"Image features shape: {img_feat.shape}")
                logger.info(f"Heatmap features shape: {heatmap_feat.shape}")
                logger.info(f"Attribute features shape: {attr_feat.shape}")
                logger.info(f"Fused features shape: {fused_feat.shape}")
                self.feature_shapes_logged = True  # Prevent repeated logging

            # Outputs
            category_logits = self.category_classifier(fused_feat)  # [batch_size, num_categories]
            category_type_logits = self.category_type_classifier(fused_feat)  # [batch_size, num_category_types]
            attribute_preds = self.attribute_predictor(fused_feat)  # [batch_size, num_attributes]

            outputs = {
                'category_logits': category_logits,
                'category_probs': F.softmax(category_logits, dim=1),  # Softmax for class probabilities
                'category_type_logits': category_type_logits,
                'category_type_probs': F.softmax(category_type_logits, dim=1),
                'attribute_preds': attribute_preds,
                'attribute_probs': torch.sigmoid(attribute_preds),      # Sigmoid for multi-label classification
                'compatibility_score': self.compatibility_head(fused_feat) 
            }

            if return_embeddings:
                outputs['embeddings'] = fused_feat

            return outputs

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Input shapes - Image: {image.shape}, Heatmap: {heatmap.shape}, Attributes: {attributes.shape}")
            raise

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze
        logger.info(f"Image encoder backbone has been {'frozen' if freeze else 'unfrozen'}")


    def get_model_size(self) -> int:
        """Calculate and return the model size in terms of parameters"""
        return sum(p.numel() for p in self.parameters())


class FashionMultiTaskLoss(nn.Module):
    """
    Multi-task loss function for the fashion hybrid model.
    Combines:
    1. Cross-entropy loss for category classification
    2. Cross-entropy loss for category type classification
    3. Binary cross-entropy loss for attribute prediction (using BCEWithLogitsLoss)
    4. MSE loss for compatibility prediction
    """

    def __init__(self, category_weight=1.0, category_type_weight=0.5,
                 attribute_weight=0.3, compatibility_weight=0.2):
        super().__init__()
        self.category_weight = category_weight
        self.category_type_weight = category_type_weight
        self.attribute_weight = attribute_weight
        self.compatibility_weight = compatibility_weight
        

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define class weights for categories
        class_weights = torch.ones(46).to(self.device)  # Default weight of 1 for all classes
        class_weights[45] = 50.0

        self.category_type_loss_fn = nn.CrossEntropyLoss()
        self.category_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.attribute_loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
        self.compatibility_loss_fn = nn.MSELoss()  # Compatibility loss (Mean Squared Error)

       

        print(f"Multi-task loss initialized with weights: "
              f"category={category_weight}, category_type={category_type_weight}, "
              f"attribute={attribute_weight}, compatibility={compatibility_weight}")

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Model outputs from forward pass
            targets (dict): Ground truth targets containing:
                - category_labels: Category labels [batch_size]
                - category_type_labels: Category type labels [batch_size]
                - attribute_targets: Binary attribute targets [batch_size, num_attributes]
                - compatibility_targets: Compatibility scores [batch_size, 1]

        Returns:
            total_loss (torch.Tensor): Combined loss
            loss_details (dict): Individual loss components for monitoring
        """
        category_loss = self.category_loss_fn(outputs['category_logits'], targets['category_labels'])
        category_type_loss = self.category_type_loss_fn(outputs['category_type_logits'], targets['category_type_labels'])
        attribute_loss = self.attribute_loss_fn(outputs['attribute_preds'], targets['attribute_targets'])
        if 'compatibility_score' in outputs:
            compatibility_loss = self.compatibility_loss_fn(outputs['compatibility_score'], targets['compatibility_targets'])
        else:
            compatibility_loss = torch.tensor(0.0, device=self.device)


        # Combine losses
        total_loss = (
            self.category_weight * category_loss +
            self.category_type_weight * category_type_loss +
            self.attribute_weight * attribute_loss +
            self.compatibility_weight * compatibility_loss 
        )
            
        loss_details = {
            'total_loss': total_loss.item(),
            'category_loss': category_loss.item(),
            'category_type_loss': category_type_loss.item(),
            'attribute_loss': attribute_loss.item(),
            'compatibility_loss': compatibility_loss.item(),
        }

        return total_loss, loss_details

# def main():
#     """Test the FashionHybridModel by running a forward pass with dummy data."""
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Initialize model
#     model = FashionHybridModel(
#         num_categories=46,
#         num_category_types=3,
#         num_attributes=50,
#         embed_dim=256,
#         use_pretrained=True,
#         dropout_rate=0.5,
#         backbone='resnet50'
#     ).to(device)

#     # Create dummy input data
#     batch_size = 4
#     dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)  # [batch_size, 3, 224, 224]
#     dummy_heatmap = torch.randn(batch_size, 1, 224, 224).to(device)  # [batch_size, 1, 224, 224]
#     dummy_attributes = torch.randn(batch_size, 50).to(device)  # [batch_size, 50]

#     # Run forward pass
#     try:
#         outputs = model(dummy_image, dummy_heatmap, dummy_attributes)
#         logger.info("Forward pass successful!")
#         for key, value in outputs.items():
#             logger.info(f"{key}: Shape={value.shape}")
#     except Exception as e:
#         logger.error(f"Error during forward pass: {str(e)}")


# if __name__ == "__main__":
#     main()



# import argparse
# import logging
# import os
# from datetime import datetime

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
# from torch.amp import GradScaler, autocast
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms

# from models.fashion_cnn import (FashionHybridModel, FashionMultiTaskLoss,
#                                 compute_f1_multilabel)
# from utils.fashion_dataset import FashionDataset

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger('TrainingScript')

# # Print CUDA information
# def print_cuda_info():
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     print(f"CUDA device count: {torch.cuda.device_count()}")
#     if torch.cuda.is_available():
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
#     model.train()
#     total_loss = 0
#     loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
#     total_f1 = 0
#     total_accuracy = 0
#     total_samples = 0

#     for batch_idx, batch in enumerate(train_loader):
#         # Move data to device
#         images = batch['image'].to(device, non_blocking=True)
#         heatmaps = batch['heatmap'].to(device, non_blocking=True)
#         attributes = batch['attributes'].to(device, non_blocking=True)

#         # Prepare targets
#         targets = {
#             'category_labels': batch['category_label'].to(device, non_blocking=True),
#             'category_type_labels': batch['category_type'].to(device, non_blocking=True),
#             'attribute_targets': attributes,
#             'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
#         }

#         # Forward pass with mixed precision
#         optimizer.zero_grad()
#         with torch.amp.autocast(device_type='cuda'):
#             outputs = model(images, heatmaps, attributes)
#             loss, loss_dict = criterion(outputs, targets)
            
#         # Compute F1 score
#         f1 = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
#         accuracy = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

#         # Backward pass with gradient clipping
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
#         scaler.step(optimizer)
#         scaler.update()

#         # Update metrics
#         total_loss += loss.item()
#         total_f1 += f1
#         total_accuracy += accuracy
#         total_samples += 1
#         for k, v in loss_dict.items():
#             if k != 'total_loss':
#                 loss_components[k.replace('_loss', '')] += v

#         if batch_idx % 10 == 0:
#             print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
#                   f'Loss: {loss.item():.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')

#     # Calculate averages
#     avg_loss = total_loss / len(train_loader)
#     avg_f1 = total_f1 / total_samples
#     avg_accuracy = total_accuracy / total_samples
#     avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

#     return avg_loss, avg_f1, avg_accuracy, avg_components

# def validate(model, val_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct_category = 0
#     correct_type = 0
#     total = 0
#     total_f1 = 0
#     total_accuracy = 0
#     total_samples = 0

#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch['image'].to(device, non_blocking=True)
#             heatmaps = batch['heatmap'].to(device, non_blocking=True)
#             attributes = batch['attributes'].to(device, non_blocking=True)

#             targets = {
#                 'category_labels': batch['category_label'].to(device, non_blocking=True),
#                 'category_type_labels': batch['category_type'].to(device, non_blocking=True),
#                 'attribute_targets': attributes,
#                 'compatibility_targets': torch.ones(images.size(0), 1).to(device, non_blocking=True)
#             }

#             outputs = model(images, heatmaps, attributes)
#             loss, _ = criterion(outputs, targets)

#             # Calculate accuracy
#             _, predicted_category = outputs['category_logits'].max(1)
#             _, predicted_type = outputs['category_type_logits'].max(1)
#             total += targets['category_labels'].size(0)
#             correct_category += predicted_category.eq(targets['category_labels']).sum().item()
#             correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()

#             total_loss += loss.item()

#             # Compute F1 score and accuracy
#             f1 = compute_f1_multilabel(outputs['category_probs'], targets['category_labels'])
#             accuracy = (outputs['category_logits'].argmax(dim=1) == targets['category_labels']).float().mean()

#             total_f1 += f1
#             total_accuracy += accuracy
#             total_samples += 1

#     avg_loss = total_loss / len(val_loader)
#     category_acc = 100. * correct_category / total
#     type_acc = 100. * correct_type / total
#     avg_f1 = total_f1 / total_samples
#     avg_accuracy = total_accuracy / total_samples

#     return avg_loss, category_acc, type_acc, avg_f1, avg_accuracy

# def main(args):
#     # Print CUDA information
#     print_cuda_info()

#     # empty the cache
#     torch.cuda.empty_cache()

#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')

#     # Create output directory in Google Drive
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
#     os.makedirs(output_dir, exist_ok=True)

#     # Initialize tensorboard
#     writer = SummaryWriter(output_dir)

#     # Define transforms with augmentation for training
#     train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Load datasets
#     train_dataset = FashionDataset(
#         metadata_path=args.train_metadata,
#         cropped_images_dir=args.images_dir,
#         heatmaps_dir=args.heatmaps_dir,
#         transform=train_transform,
#         use_cache=True,
#         cache_size=100,
#         validate_files=True,
#     )

#     val_dataset = FashionDataset(
#         metadata_path=args.val_metadata,
#         cropped_images_dir=args.images_dir,
#         heatmaps_dir=args.heatmaps_dir,
#         transform=train_transform,
#         use_cache=True,
#         cache_size=100,
#         validate_files=True,
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         persistent_workers=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         persistent_workers=True,
#     )

#     # Initialize model
#     model = FashionHybridModel(
#         num_categories=args.num_categories,
#         num_category_types=args.num_category_types,
#         num_attributes=args.num_attributes,
#         backbone=args.backbone
#     )

#     # Use DataParallel if multiple GPUs are available
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs!")
#         model = torch.nn.DataParallel(model)
#     model = model.to(device)

#     # Initialize criterion and optimizer
#     criterion = FashionMultiTaskLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5
#     )

#     # Mixed precision training
#     scaler = GradScaler()

#     # Training loop with early stopping
#     best_val_loss = float('inf')
#     patience = 5  # Number of epochs to wait for improvement
#     epochs_without_improvement = 0

#     for epoch in range(args.epochs):
#         # Train
#         train_loss, train_f1, train_accuracy, train_components = train_epoch(
#             model, train_loader, criterion, optimizer, device, epoch, scaler
#         )

#         # Validate
#         val_loss, category_acc, type_acc, val_f1, val_accuracy = validate(model, val_loader, criterion, device)

#         # Log metrics
#         print(f'Epoch: {epoch}')
#         print(f'Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}')
#         print(f'Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}')
#         print(f'Category Accuracy: {category_acc:.2f}%')
#         print(f'Type Accuracy: {type_acc:.2f}%')

#         # Tensorboard logging
#         writer.add_scalar('Loss/train', train_loss, epoch)
#         writer.add_scalar('Loss/val', val_loss, epoch)
#         writer.add_scalar('Accuracy/train', train_accuracy, epoch)
#         writer.add_scalar('Accuracy/val', val_accuracy, epoch)
#         writer.add_scalar('F1/train', train_f1, epoch)
#         writer.add_scalar('F1/val', val_f1, epoch)

#         # Save model for each epoch
#         torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_loss,
#             }, os.path.join("/content/drive/MyDrive/Runs/Epochs", f'epoch_{timestamp}_{epoch}.pth'))

#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_without_improvement = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_loss,
#             }, os.path.join(output_dir, 'best_model.pth'))
#         else:
#             epochs_without_improvement += 1

#         # Early stopping
#         if epochs_without_improvement >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break

#         # Update learning rate
#         scheduler.step(val_loss)

#     writer.close()

# # Define the args object
# class Args:
#     def __init__(self):
#         self.train_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_train.csv'
#         self.val_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_val.csv'
#         self.images_dir = '/content/cropped_images1'
#         self.heatmaps_dir = '/content/heatmaps1'
#         self.backbone = 'resnet50'
#         self.num_categories = 50
#         self.num_category_types = 4
#         self.num_attributes = 50
#         self.batch_size = 512
#         self.epochs = 2
#         self.learning_rate = 0.001
#         self.num_workers = 2

# # Create the args object
# args = Args()

# # Run the script
# if __name__ == '__main__':
#     main(args)


