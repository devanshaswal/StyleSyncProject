#Model
import logging
from typing import Dict

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


def compute_f1_multiclass(preds, targets, average='macro'):
    preds = preds.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average=average, zero_division=1)


def compute_f1_multilabel(preds, targets, average='macro'):
    preds = (preds > 0.5).int().cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average=average, zero_division=1)



class FashionHybridModel(nn.Module):
    def __init__(self, num_categories=46, num_category_types=3, num_attributes=50,
                 embed_dim=256, use_pretrained=True, dropout_rate=0.5, backbone='resnet50'):
        super().__init__()
        logger.info(f"Initializing FashionHybridModel with {backbone} backbone")

        # Initialize backbone
        self.backbone = backbone
        self.image_feature_dim = self._init_image_encoder(use_pretrained)

        # Heatmap encoder
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.heatmap_feature_dim = 64

        # Attribute encoder
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
        self.attribute_feature_dim = 64

        total_feature_dim = self.image_feature_dim + self.heatmap_feature_dim + self.attribute_feature_dim
        logger.info(f"Total feature dimension after concatenation: {total_feature_dim}")

        # Auxiliary Classifiers for Deep Supervision
        self.image_aux_classifier = nn.Linear(self.image_feature_dim, num_categories)
        self.heatmap_aux_classifier = nn.Linear(self.heatmap_feature_dim, num_category_types)
        self.attribute_aux_classifier = nn.Linear(self.attribute_feature_dim, num_attributes)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7)
        )

        # Classification heads
        self.category_classifier = nn.Linear(embed_dim, num_categories)
        self.category_type_classifier = nn.Linear(embed_dim, num_category_types)
        self.attribute_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, num_attributes)
        )

        self.compatibility_head = nn.Linear(embed_dim, 1)

        logger.info("Model initialization complete")

    def _init_image_encoder(self, use_pretrained):
        if self.backbone == 'resnet50':
            model = models.resnet50(weights="IMAGENET1K_V1" if use_pretrained else None)
            feature_dim = 2048
        else:
            logger.warning(f"Unknown backbone {self.backbone}, using resnet50")
            model = models.resnet50(weights="IMAGENET1K_V1" if use_pretrained else None)
            feature_dim = 2048

        self.image_encoder = nn.Sequential(*list(model.children())[:-1])
        return feature_dim

    def forward(self, image, heatmap, attributes):
        img_feat = torch.flatten(self.image_encoder(image), 1)
        heatmap_feat = self.heatmap_encoder(heatmap).view(image.size(0), -1)
        attr_feat = self.attribute_encoder(attributes)

        fused_feat = self.fusion(torch.cat([img_feat, heatmap_feat, attr_feat], dim=1))

        # Auxiliary Outputs (Deep Supervision)
        image_aux_logits = self.image_aux_classifier(img_feat)
        heatmap_aux_logits = self.heatmap_aux_classifier(heatmap_feat)
        attribute_aux_logits = self.attribute_aux_classifier(attr_feat)


        # Main Outputs
        category_logits = self.category_classifier(fused_feat)  # [batch_size, num_categories]
        category_type_logits = self.category_type_classifier(fused_feat)  # [batch_size, num_category_types]
        attribute_preds = self.attribute_predictor(fused_feat)  # [batch_size, num_attributes]

        outputs = {
            'category_logits': category_logits,
            'category_probs': F.softmax(category_logits, dim=1),
            'category_type_logits': category_type_logits,
            'category_type_probs': F.softmax(category_type_logits, dim=1),
            'attribute_preds': attribute_preds,
            'attribute_probs': torch.sigmoid(attribute_preds),
            'compatibility_score': self.compatibility_head(fused_feat),
            'image_aux_logits': image_aux_logits,
            'heatmap_aux_logits': heatmap_aux_logits,
            'attribute_aux_logits': attribute_aux_logits
        }
        return outputs

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze

        trainable_params = sum(p.requires_grad for p in self.image_encoder.parameters())
        logger.info(f"Image encoder backbone has been {'frozen' if freeze else 'unfrozen'}. Trainable params: {trainable_params}")



class FashionMultiTaskLoss(nn.Module):
    def __init__(self, category_weight=1.0, category_type_weight=0.5,
                 attribute_weight=0.3, compatibility_weight=0.2, aux_weight=0.3,
                 compatibility_loss_type='mse', total_epochs=30):  # total_epochs as a parameter
        super().__init__()
        self.category_weight = category_weight
        self.category_type_weight = category_type_weight
        self.attribute_weight = attribute_weight
        self.compatibility_weight = compatibility_weight
        self.aux_weight = aux_weight
        self.total_epochs = total_epochs  # store total_epochs

        self.category_loss_fn = nn.CrossEntropyLoss()
        self.category_type_loss_fn = nn.CrossEntropyLoss()
        self.attribute_loss_fn = nn.BCEWithLogitsLoss()
        if compatibility_loss_type == 'bce':
            self.compatibility_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.compatibility_loss_fn = nn.MSELoss()

    def forward(self, outputs, targets, epoch):
        category_loss = self.category_loss_fn(outputs['category_logits'], targets['category_labels'])
        category_type_loss = self.category_type_loss_fn(outputs['category_type_logits'], targets['category_type_labels'])
        attribute_loss = self.attribute_loss_fn(outputs['attribute_preds'], targets['attribute_targets'])
        compatibility_loss = self.compatibility_loss_fn(outputs['compatibility_score'], targets['compatibility_targets'])

        # Auxiliary losses (Deep Supervision)
        image_aux_loss = self.category_loss_fn(outputs['image_aux_logits'], targets['category_labels'])
        heatmap_aux_loss = self.category_type_loss_fn(outputs['heatmap_aux_logits'], targets['category_type_labels'])
        attribute_aux_loss = self.attribute_loss_fn(outputs['attribute_aux_logits'], targets['attribute_targets'])
        total_aux_loss = image_aux_loss + heatmap_aux_loss + attribute_aux_loss

        # Dynamic auxiliary loss scaling using the stored total_epochs
        scaling_factor = max(0.1, 1.0 - (epoch / self.total_epochs))
        aux_weight = self.aux_weight * scaling_factor  # Dynamic aux loss weight

        total_loss = (
            self.category_weight * category_loss +
            self.category_type_weight * category_type_loss +
            self.attribute_weight * attribute_loss +
            self.compatibility_weight * compatibility_loss +
            aux_weight * total_aux_loss
        )

        loss_dict = {
            'category_loss': category_loss.item(),
            'category_type_loss': category_type_loss.item(),
            'attribute_loss': attribute_loss.item(),
            'compatibility_loss': compatibility_loss.item(),
            'image_aux_loss': image_aux_loss.item(),
            'heatmap_aux_loss': heatmap_aux_loss.item(),
            'attribute_aux_loss': attribute_aux_loss.item(),
            'total_aux_loss': total_aux_loss.item()
        }

        return total_loss, loss_dict