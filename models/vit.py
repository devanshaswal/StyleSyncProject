import logging
import math
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.auto_augment import rand_augment_transform
from timm.models.layers import DropPath, trunc_normal_
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.checkpoint import checkpoint

# Set up logging
logger = logging.getLogger("FashionViTModel")

class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss with class balancing and label smoothing support
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        targets = F.one_hot(targets, num_classes=logits.size(-1))
        targets = (1 - self.label_smoothing) * targets + self.label_smoothing / logits.size(-1)
        ce_loss = -torch.sum(targets * log_probs, dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class FocalLossMultiLabel(nn.Module):
    """
    Enhanced Multi-label Focal Loss with per-group weighting
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
        else:
            self.pos_weight = None
        self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        alpha = self.alpha.to(bce_loss.device).type_as(bce_loss)
        if alpha.ndim == 1 and alpha.shape[0] == logits.shape[1]:
            alpha = alpha.unsqueeze(0)
        focal_loss = alpha * ((1.0 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class CrossModalAttention(nn.Module):
    """
    Enhanced Cross-Modal Attention with residual connections
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.heat_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, img_feat, heat_feat):
        proj_heat = self.heat_proj(heat_feat).unsqueeze(1)
        proj_img = self.img_proj(img_feat)
        
        attn_out, _ = self.attention(
            query=proj_img,
            key=proj_heat,
            value=proj_heat
        )
        return self.norm(img_feat + self.dropout(attn_out))

class TransformerBlock(nn.Module):
    """
    Enhanced Transformer Block with stochastic depth
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GroupedAttributeHead(nn.Module):
    """
    Enhanced Grouped Attribute Head with shared feature extraction
    """
    def __init__(self, input_dim, group_sizes):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.sub_heads = nn.ModuleDict({
            group: nn.Sequential(
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 4, size))
            for group, size in group_sizes.items()
        })

    def forward(self, x):
        shared = self.shared_features(x)
        return {group: head(shared) for group, head in self.sub_heads.items()}

class FashionViTModel(nn.Module):
    """
    Enhanced FashionViT Model with improved architecture and debugging
    """
    def __init__(self, num_categories, num_category_types, group_sizes,
                 vit_name='vit_base_patch16_224', use_pretrained=True,
                 embed_dim=768, fusion_dim=768, dropout_rate=0.1,
                 drop_path_rate=0.1, enable_deep_supervision=True):
        super().__init__()
        logger.info(f"Initializing FashionViTModel with {vit_name}")
        
        # Vision Transformer backbone
        self.vit = timm.create_model(
            vit_name,
            pretrained=use_pretrained,
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            img_size=224
        )
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        
        # Heatmap processing
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        # Cross-modal fusion
        self.cross_attn = CrossModalAttention(embed_dim, num_heads=8)
        self.transformer_block = TransformerBlock(embed_dim, num_heads=8)
        
        # Classification heads
        self.category_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, num_categories))
        
        self.category_type_classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_category_types))
        
        self.attribute_predictor = GroupedAttributeHead(fusion_dim, group_sizes)
        
        # Deep supervision
        if enable_deep_supervision:
            self.image_aux_classifier = nn.Linear(embed_dim, num_categories)
            self.heatmap_aux_classifier = nn.Linear(embed_dim, num_category_types)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, image, heatmap):
        # Feature extraction
        img_feat = self.vit.forward_features(image)
        heat_feat = self.heatmap_encoder(heatmap).squeeze([2, 3])
        

        # Cross-modal fusion with gradient checkpointing
        fused_feat = checkpoint(self.cross_attn, img_feat, heat_feat)
        fused_feat = self.transformer_block(fused_feat)
        
        # Classification
        outputs = {
            'category_logits': self.category_classifier(fused_feat[:, 0]),
            'category_type_logits': self.category_type_classifier(fused_feat[:, 0]),
            'attribute_preds': self.attribute_predictor(fused_feat[:, 0])
        }
        
        # Deep supervision
        if self.enable_deep_supervision:
            outputs['image_aux_logits'] = self.image_aux_classifier(img_feat[:, 0])
            outputs['heatmap_aux_logits'] = self.heatmap_aux_classifier(heat_feat)

        
        return outputs

    def freeze_backbone(self, freeze=True):
        for param in self.vit.parameters():
            param.requires_grad = not freeze
        logger.info(f"ViT backbone {'frozen' if freeze else 'unfrozen'}")

# ----------------------------
# Enhanced Multi-Task Loss
# ----------------------------
class FashionMultiTaskLoss(nn.Module):
    def __init__(self, category_weight, category_type_weight, attribute_weight, aux_weight, total_epochs, attr_groups):
        """
        Args:
            category_weight (float): Weight for category loss
            category_type_weight (float): Weight for category type loss
            attribute_weight (float): Weight for attribute loss
            aux_weight (float): Weight for auxiliary loss
            total_epochs (int): Total epochs for training
            attr_groups (dict): Dictionary mapping attribute group names to their indices
        """
        super().__init__()
        self.category_weight = category_weight
        self.category_type_weight = category_type_weight
        self.attribute_weight = attribute_weight
        self.aux_weight = aux_weight
        self.total_epochs = total_epochs
        self.attribute_groups = attr_groups  #  Now explicitly defined

        # Loss functions
        self.category_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.category_type_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.attribute_loss_fn = FocalLossMultiLabel(alpha=0.25, gamma=2.0)

    def forward(self, outputs, targets, epoch):

        # Category and category type losses
        loss_cat = self.category_loss_fn(outputs['category_logits'], targets['category_labels'])
        loss_cat_type = self.category_type_loss_fn(outputs['category_type_logits'], targets['category_type_labels'])

        # Grouped attribute loss calculation
        loss_attr = 0.0
        num_attr_groups = 0

        for group_name, logits in outputs['attribute_preds'].items():
            # Get the indices of attributes belonging to the current group
            group_indices = self.attribute_groups.get(group_name, [])

            if len(group_indices) > 0:  # Ensure the group has valid attribute indices
                group_targets = targets['attribute_targets'][:, group_indices]  # Extract only valid attributes
                group_loss = self.attribute_loss_fn(logits, group_targets)
                loss_attr += group_loss
                num_attr_groups += 1
            else:
                logger.warning(f"Skipping attribute group {group_name} due to empty index list.")

        # Average attribute loss across groups
        if num_attr_groups > 0:
            loss_attr = loss_attr / num_attr_groups

        # Deep supervision losses
        aux_loss = 0.0
        if 'image_aux_logits' in outputs:
            aux_loss += self.category_loss_fn(outputs['image_aux_logits'], targets['category_labels'])
        if 'heatmap_aux_logits' in outputs:
            aux_loss += self.category_type_loss_fn(outputs['heatmap_aux_logits'], targets['category_type_labels'])

        # Dynamic auxiliary loss weighting
        aux_weight = self.aux_weight * max(0.1, 1.0 - (epoch / self.total_epochs))

        # Total loss calculation
        total_loss = (
            self.category_weight * loss_cat +
            self.category_type_weight * loss_cat_type +
            self.attribute_weight * loss_attr +
            aux_weight * aux_loss
        )

        # Loss logging dictionary
        loss_dict = {
            'category_loss': loss_cat.item(),
            'category_type_loss': loss_cat_type.item(),
            'attribute_loss': loss_attr.item(),
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            'total_loss': total_loss.item()
        }

        # Add per-group attribute losses for detailed monitoring
        for group_name, logits in outputs['attribute_preds'].items():
            group_indices = self.attribute_groups[group_name]  # Get index list from predefined groups
            group_targets = targets['attribute_targets'][:, group_indices]  # Slice relevant attributes

            group_loss = self.attribute_loss_fn(logits, group_targets)
            loss_dict[f'attr_{group_name}_loss'] = group_loss.item()  # Store each group’s loss

        return total_loss, loss_dict

