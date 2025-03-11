#MODEL_SCRIPT
from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # Import necessary for softmax/sigmoid
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


def compute_f1_multilabel(preds, targets, threshold=0.5):
    """Computes F1-score for multi-label classification."""
    preds = (preds > threshold).int().cpu().numpy()  # Apply threshold to predictions
    targets = targets.cpu().numpy()  # Convert targets to numpy for F1 computation
    return f1_score(targets, preds, average='macro')  # Compute F1-score


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
        print(f"Attribute encoder initialized with output dimension: {self.attribute_feature_dim}")

        # Total feature dimension after concatenation
        total_feature_dim = self.image_feature_dim + self.heatmap_feature_dim + self.attribute_feature_dim
        print(f"Total feature dimension after concatenation: {total_feature_dim}")

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

        print("Model initialization complete")

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

        print(f"Initialized {self.backbone} backbone with feature dimension: {feature_dim}")
        return feature_dim


    # New method
    def compute_accuracy(self, preds, targets, topk=(1,)):
        """Computes the top-k accuracy for the given predictions and targets."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)  # Get top-k predictions
            pred = pred.t()  # Transpose for comparison
            correct = pred.eq(targets.view(1, -1).expand_as(pred))  # Compare with ground truth

            accs = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                accs.append(correct_k.mul_(100.0 / batch_size))  # Convert to percentage

            return accs  # Returns top-1, top-5 accuracies



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
        batch_size = image.size(0)

        try:
            # Image features
            img_feat = self.image_encoder(image)  # [batch_size, channels, 1, 1] or [batch_size, channels, H, W]
            img_feat = torch.flatten(img_feat, 1)  # [batch_size, image_feature_dim]

            # Heatmap features
            heatmap_feat = self.heatmap_encoder(heatmap)  # [batch_size, heatmap_feature_dim, 1, 1]
            heatmap_feat = heatmap_feat.view(batch_size, -1)  # [batch_size, heatmap_feature_dim]

            # Attribute features
            attr_feat = self.attribute_encoder(attributes)  # [batch_size, attribute_feature_dim]

            # Concatenate features
            fused_feat = torch.cat([img_feat, heatmap_feat, attr_feat], dim=1)
            fused_feat = self.fusion(fused_feat)  # [batch_size, embed_dim]

            # Outputs
            category_logits = self.category_classifier(fused_feat)  # [batch_size, num_categories]
            category_type_logits = self.category_type_classifier(fused_feat)  # [batch_size, num_category_types]
            attribute_preds = self.attribute_predictor(fused_feat)  # [batch_size, num_attributes]
            compatibility_score = self.compatibility_head(fused_feat)  # [batch_size, 1]

            outputs = {
                'category_logits': category_logits,
                'category_probs': F.softmax(category_logits, dim=1),  # Softmax for class probabilities
                'category_type_logits': category_type_logits,
                'category_type_probs': F.softmax(category_type_logits, dim=1),
                'attribute_preds': attribute_preds,   # Use the correct variable       # This ensures the output is in [0, 1] range torch.sigmoid(attribute_preds)
                'attribute_probs': torch.sigmoid(attribute_preds),  # Sigmoid for multi-label classification
                'compatibility_score': compatibility_score
            }

            if return_embeddings:
                outputs['embeddings'] = fused_feat

            return outputs

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Batch sizes - Image: {image.size()}, Heatmap: {heatmap.size()}, Attributes: {attributes.size()}")
            raise

    def freeze_backbone(self, freeze: bool = True) -> None:
       """
        Freezes or unfreezes the image encoder backbone parameters (i.e., ResNet or EfficientNet layers).

        Args:
            freeze (bool): If True, freeze the backbone parameters (set requires_grad=False).
                            If False, unfreeze the backbone parameters (set requires_grad=True).
        """
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze
        print(f"Image encoder backbone has been {'frozen' if freeze else 'unfrozen'}")

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

        self.category_loss_fn = nn.CrossEntropyLoss()
        self.category_type_loss_fn = nn.CrossEntropyLoss()
        self.attribute_loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
        self.compatibility_loss_fn = nn.MSELoss()

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
        compatibility_loss = self.compatibility_loss_fn(outputs['compatibility_score'], targets['compatibility_targets'])

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
            'compatibility_loss': compatibility_loss.item()
        }

        return total_loss, loss_details