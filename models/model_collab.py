import logging
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_collab import FashionDataset

# Setup logging - reduced for Colab environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console-only logging for Colab
    ]
)
logger = logging.getLogger('FashionHybridModel')

class FashionHybridModel(nn.Module):
    """
    Memory-optimized hybrid CNN model for fashion recommendation.
    Processes:
    - Cropped images (224x224 RGB)
    - Landmark heatmaps (224x224 single-channel)
    - Attribute metadata
    """
    
    def __init__(self, num_categories=46, num_category_types=3, num_attributes=50, embed_dim=128,
                 use_pretrained=True, dropout_rate=0.3, backbone='resnet18'):
        """
        Args:
            num_categories (int): Number of clothing categories
            num_category_types (int): Number of category types
            num_attributes (int): Number of attributes
            embed_dim (int): Dimension of the fused feature embedding (reduced for memory)
            use_pretrained (bool): Whether to use pretrained weights for the backbone
            dropout_rate (float): Dropout rate for regularization
            backbone (str): CNN backbone architecture ('resnet18', 'resnet50', 'efficientnet_b0')
        """
        super().__init__()
        logger.info(f"Initializing memory-optimized FashionHybridModel with {backbone} backbone")
        
        # Initialize backbone - lighter options for memory constraints
        self.backbone = backbone
        self.image_feature_dim = self._init_image_encoder(use_pretrained)
        
        # Lighter heatmap encoder
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Reduced filters: 32->16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 112, 112]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced filters: 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 56, 56]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced filters: 128->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 28, 28]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Reduced filters: 256->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # [128, 1, 1]
        )
        self.heatmap_feature_dim = 128  # Reduced from 256
        
        # Optimized attribute encoder
        self.attribute_encoder = nn.Sequential(
            nn.Linear(num_attributes, 64),  # Reduced dimension: 128->64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),  # Reduced dimension: 64->32
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.attribute_feature_dim = 32  # Reduced from 64
        
        # Total feature dimension after concatenation
        total_feature_dim = self.image_feature_dim + self.heatmap_feature_dim + self.attribute_feature_dim
        
        # Fusion layer (optimized)
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Classification heads (optimized)
        # 1. Category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),  # Reduced: 512->256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_categories)
        )
        
        # 2. Category type classifier
        self.category_type_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),  # Reduced: 128->64
            nn.ReLU(inplace=True),
            nn.Linear(64, num_category_types)
        )
        
        # 3. Attribute predictor - simplified
        self.attribute_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),  # Reduced: 256->128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_attributes),
            nn.Sigmoid()
        )
        
        # 4. Compatibility head - simplified
        self.compatibility_head = nn.Sequential(
            nn.Linear(embed_dim, 64),  # Reduced: 128->64
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Memory-optimized model initialized with {self.get_model_size()} parameters")
    
    def _init_image_encoder(self, use_pretrained):
        """Initialize a memory-efficient image encoder backbone"""
        feature_dim = 0
        
        # Use lighter backbones for memory efficiency
        if self.backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 512  # Much smaller than resnet50's 2048
        elif self.backbone == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 512
        elif self.backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 1280
        elif self.backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 1280  # Smaller than b3's 1536
        elif self.backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 2048
        else:
            logger.warning(f"Unknown backbone {self.backbone}, falling back to resnet18")
            model = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
            feature_dim = 512
            
        # Remove the final fully connected layer
        if 'resnet' in self.backbone or 'resnext' in self.backbone:
            self.image_encoder = nn.Sequential(*list(model.children())[:-1])
        elif 'mobilenet' in self.backbone:
            self.image_encoder = model.features
        else:  # EfficientNet
            self.image_encoder = model.features
            
        logger.info(f"Using {self.backbone} backbone with {feature_dim} output dimensions")
        return feature_dim
        
    def forward(self, image, heatmap, attributes, return_embeddings=False):
        """
        Forward pass with memory optimization
        """
        batch_size = image.size(0)
        
        try:
            # Process in stages with torch.no_grad() for feature extraction when possible
            # Image features
            img_feat = self.image_encoder(image)
            img_feat = torch.flatten(img_feat, 1)
            
            # Heatmap features
            heatmap_feat = self.heatmap_encoder(heatmap)
            heatmap_feat = heatmap_feat.view(batch_size, -1)
            
            # Attribute features
            attr_feat = self.attribute_encoder(attributes)
            
            # Concatenate features
            fused_feat = torch.cat([img_feat, heatmap_feat, attr_feat], dim=1)
            fused_feat = self.fusion(fused_feat)
            
            # Compute outputs
            category_logits = self.category_classifier(fused_feat)
            category_type_logits = self.category_type_classifier(fused_feat)
            attribute_preds = self.attribute_predictor(fused_feat)
            compatibility_score = self.compatibility_head(fused_feat)
            
            outputs = {
                'category_logits': category_logits,
                'category_type_logits': category_type_logits,
                'attribute_preds': attribute_preds,
                'compatibility_score': compatibility_score
            }
            
            if return_embeddings:
                outputs['embeddings'] = fused_feat
                
            return outputs
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            # More detailed error reporting
            shapes = {
                'image': tuple(image.shape),
                'heatmap': tuple(heatmap.shape),
                'attributes': tuple(attributes.shape)
            }
            logger.error(f"Input shapes: {shapes}")
            raise
    
    def freeze_backbone(self, freeze=True, layers=None):
        """
        Freeze or unfreeze backbone with granular control
        Args:
            freeze (bool): Whether to freeze or unfreeze
            layers (list, optional): List of layer indices to freeze/unfreeze (resnet only)
        """
        if layers is None:
            # Freeze/unfreeze entire backbone
            for param in self.image_encoder.parameters():
                param.requires_grad = not freeze
            logger.info(f"Image encoder backbone {'frozen' if freeze else 'unfrozen'}")
        else:
            # Only for resnet models
            if 'resnet' in self.backbone:
                # Individual layer freezing (useful for fine-tuning with limited memory)
                children = list(self.image_encoder.children())
                for idx in layers:
                    if 0 <= idx < len(children):
                        for param in children[idx].parameters():
                            param.requires_grad = not freeze
                logger.info(f"Specified layers {layers} {'frozen' if freeze else 'unfrozen'}")
    
    def get_model_size(self):
        """Calculate and return the model size in MB"""
        num_params = sum(p.numel() for p in self.parameters())
        size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
        return {"params": num_params, "size_mb": size_mb}

    def get_memory_usage(self, input_shape=(1, 3, 224, 224)):
        """Estimate forward pass memory usage"""
        try:
            from torch.utils.hooks import RemovableHandle
            
            # Track max memory allocation for a single forward pass
            mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            
            # Create dummy inputs
            image = torch.zeros(input_shape, device=next(self.parameters()).device)
            batch_size = input_shape[0]
            heatmap = torch.zeros((batch_size, 1, 224, 224), device=image.device)
            attributes = torch.zeros((batch_size, 50), device=image.device)
            
            # Forward pass
            with torch.no_grad():
                _ = self.forward(image, heatmap, attributes)
            
            mem_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (mem_after - mem_before) / (1024 * 1024)  # MB
            
            return {"forward_pass_mb": memory_used}
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            return {"forward_pass_mb": "Error estimating"}


# Memory-optimized loss function
class FashionMultiTaskLoss(nn.Module):
    """
    Memory-efficient multi-task loss function for the fashion hybrid model.
    """
    
    def __init__(self, category_weight=1.0, category_type_weight=0.5, 
                 attribute_weight=0.3, compatibility_weight=0.2,
                 label_smoothing=0.1):
        super().__init__()
        self.category_weight = category_weight
        self.category_type_weight = category_type_weight
        self.attribute_weight = attribute_weight
        self.compatibility_weight = compatibility_weight
        
        # Add label smoothing for better regularization
        self.category_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.category_type_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.attribute_loss_fn = nn.BCELoss()
        self.compatibility_loss_fn = nn.MSELoss()
        
        logger.info(f"Multi-task loss initialized with weights: "
                   f"category={category_weight}, type={category_type_weight}, "
                   f"attribute={attribute_weight}, compatibility={compatibility_weight}")
    
    def forward(self, outputs, targets):
        """Compute loss with memory optimization"""
        # Calculate each loss component
        category_loss = self.category_loss_fn(outputs['category_logits'], targets['category_labels'])
        category_type_loss = self.category_type_loss_fn(outputs['category_type_logits'], targets['category_type_labels'])
        attribute_loss = self.attribute_loss_fn(outputs['attribute_preds'], targets['attribute_targets'])
        compatibility_loss = self.compatibility_loss_fn(outputs['compatibility_score'], targets['compatibility_targets'])
        
        # Combine losses
        total_loss = (
            self.category_weight * category_loss +
            self.category_type_weight * category_type_loss +
            self.attribute_weight * attribute_loss +
            self.compatibility_weight * compatibility_loss
        )
        
        # Return detailed loss dict only when needed (e.g., for logging)
        if random.random() < 0.1:  # Only compute full details 10% of the time to save memory
            loss_details = {
                'total': total_loss.item(),
                'category': category_loss.item(),
                'category_type': category_type_loss.item(),
                'attribute': attribute_loss.item(),
                'compatibility': compatibility_loss.item()
            }
            return total_loss, loss_details
        
        # For 90% of batches, return minimal info
        return total_loss, {'total': total_loss.item()}


# Batch helper for memory optimization
def process_batch_with_mixed_precision(model, batch, loss_fn, optimizer=None, 
                                      scaler=None, is_training=True, device='cuda'):
    """
    Process a batch with optional mixed precision for memory efficiency
    """
    # Move data to device
    image = batch['image'].to(device)
    heatmap = batch['heatmap'].to(device)
    attributes = batch['attributes'].to(device)
    
    # Prepare targets
    targets = {
        'category_labels': batch['category_label'].to(device),
        'category_type_labels': batch['category_type'].to(device),
        'attribute_targets': attributes,  # Using input attributes as targets
        'compatibility_targets': torch.ones(image.size(0), 1).to(device)  # Dummy values
    }
    
    # Mixed precision for memory efficiency if scaler is provided
    if is_training:
        optimizer.zero_grad()
        
        if scaler is not None:  # Use mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(image, heatmap, attributes)
                loss, loss_details = loss_fn(outputs, targets)
            
            # Scale and backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular precision
            outputs = model(image, heatmap, attributes)
            loss, loss_details = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    else:
        # Evaluation mode
        with torch.no_grad():
            outputs = model(image, heatmap, attributes)
            loss, loss_details = loss_fn(outputs, targets)
    
    return outputs, loss.item(), loss_details


# Add import for batch processing helper
# import random

# # Example usage
# if __name__ == "__main__":
#     # Path configuration for Colab
#     metadata_path = "/content/drive/MyDrive/stylesync/metadata_updated.csv"
#     cropped_images_dir = '/content/cropped_images/cropped_images'
#     heatmaps_dir = '/content/heatmaps/heatmaps'
    
#     # Initialize model with more efficient backbone
#     model = FashionHybridModel(
#         num_categories=46,
#         num_category_types=3,
#         num_attributes=50,
#         embed_dim=128,  # Reduced from 256
#         use_pretrained=True,
#         dropout_rate=0.3,  # Reduced dropout for faster convergence
#         backbone='resnet18'  # Lighter backbone
#     )
    
#     # Print model memory usage
#     print(f"Model size: {model.get_model_size()}")
    
#     # Example for creating a memory-efficient training loop
#     from torch.utils.data import DataLoader
#     import torch.optim as optim
#     from torch.cuda.amp import GradScaler
    
    
    
#     # Create dataset with minimal memory footprint
#     dataset = FashionDataset(
#         metadata_path=metadata_path,
#         cropped_images_dir=cropped_images_dir,
#         heatmaps_dir=heatmaps_dir,
#         use_cache=True,
#         cache_size=50,
#         validate_files=False
#     )
    
#     # Create dataloader with smaller batch size
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=8,  # Smaller batch size for memory efficiency
#         shuffle=True,
#         num_workers=0,  # No extra workers for Colab
#         pin_memory=False  # Disable pin_memory to reduce memory usage
#     )
    
#     # Init optimizer with memory-efficient settings
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=1e-4,
#         weight_decay=1e-4,
#         betas=(0.9, 0.999),
#         eps=1e-8
#     )
    
#     # Loss function
#     criterion = FashionMultiTaskLoss()
    
#     # Use mixed precision for memory efficiency
#     scaler = GradScaler()
    
#     # Example training loop (minimal implementation)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     # Just showing how to use the model with the dataset
#     print("Example model setup complete")