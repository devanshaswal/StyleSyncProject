import logging
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FashionDataset')

class FashionDataset(Dataset):
    def __init__(self, metadata_path, cropped_images_dir, heatmaps_dir, 
                 attribute_groups, transform=None, use_cache=True, 
                 cache_size=100, validate_files=True):
        """
        Args:
            metadata_path (str): Path to metadata CSV file
            cropped_images_dir (str): Root directory for cropped images
            heatmaps_dir (str): Root directory for heatmap numpy files
            attribute_groups (dict): Dictionary of attribute groups
            transform (callable, optional): Optional transform to be applied
            use_cache (bool): Whether to cache loaded images and heatmaps in memory
            cache_size (int): Size of LRU cache for images and heatmaps
            validate_files (bool): Whether to validate existence of files during initialization
        """
        super().__init__()
        logger.info(f"Initializing FashionDataset with metadata: {metadata_path}")
        
        # Validate paths
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        if not os.path.exists(cropped_images_dir):
            logger.error(f"Cropped images directory not found: {cropped_images_dir}")
            raise FileNotFoundError(f"Cropped images directory not found: {cropped_images_dir}")

        if not os.path.exists(heatmaps_dir):
            logger.error(f"Heatmaps directory not found: {heatmaps_dir}")
            raise FileNotFoundError(f"Heatmaps directory not found: {heatmaps_dir}")

        # Load metadata
        try:
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

        # Store paths and configurations
        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir
        self.transform = transform or self.default_transforms()
        self.attribute_groups = attribute_groups
        self.group_to_indices = self._create_group_to_indices()
        
        # Initialize category mapping and validation
        self._initialize_category_mapping()
        
        # Custom caching
        self.use_cache = use_cache
        self.image_cache = {}
        self.heatmap_cache = {}
        self.cache_size = cache_size

        # Validate files if requested
        if validate_files:
            self.validate_dataset_files()

    def _create_group_to_indices(self):
        """Create mapping from attribute groups to column indices"""
        group_to_indices = {}
        for group, attributes in self.attribute_groups.items():
            group_to_indices[group] = [
                self.metadata.columns.get_loc(attr) for attr in attributes
            ]
        return group_to_indices

    def _initialize_category_mapping(self):
        """Initialize and validate category mapping"""
        # Remap category labels to 0-indexed
        valid_categories = [c for c in range(1, 51) if c not in [38, 45, 49, 50]]
        logger.info(f"Valid categories after filtering: {valid_categories}")

        self.metadata['original_category_label'] = self.metadata['category_label'].copy()

        self.category_mapping = {
        old_label: new_label for new_label, old_label in enumerate(sorted(valid_categories))
        }
        logger.info(f"Category mapping: {self.category_mapping}")

        # Apply mapping to category labels
        self.metadata = self.metadata[self.metadata['category_label'].isin(valid_categories)]
        self.metadata['category_label'] = self.metadata['category_label'].map(self.category_mapping)

        # Validate mapping
        assert self.metadata['category_label'].min() == 0, "Category labels should start from 0"
        assert self.metadata['category_label'].max() == 45, "Category labels should range from 0 to 45"

        # Convert category types to 0-based index
        self.metadata['category_type'] = self.metadata['category_type'] - 1

        # Debugging: Verify labels are now 0-indexed
        unique_labels = sorted(self.metadata['category_label'].unique().tolist())
        logger.info(f"Final category labels after remapping: {unique_labels}")
        print(f"Final Category Labels: {unique_labels}")  # Should show [0, 1, ..., 45]
        # Create category name mapping if available
        try:
            self.category_names = dict(zip(
                self.metadata['category_label'].unique(),
                self.metadata['category_name'].unique() if 'category_name' in self.metadata.columns
                else self.metadata['category_label'].unique()
            ))
            logger.info(f"Created category mapping with {len(self.category_names)} categories")
        except Exception as e:
            logger.warning(f"Could not create category name mapping: {str(e)}")
            self.category_names = dict(zip(
            self.metadata['category_label'].unique(),
            self.metadata['category_label'].unique()
            ))

    def __len__(self):
        return len(self.metadata)

    def _get_default_item(self):
        """Returns a default item with blank image, heatmap, and zero attributes."""
        return {
            'image': torch.zeros((3, 224, 224), dtype=torch.float32),  # Blank RGB image
            'heatmap': torch.zeros((1, 224, 224), dtype=torch.float32),  # Blank heatmap
            'attribute_targets': {
                group: torch.zeros(len(indices), dtype=torch.float32)
                for group, indices in self.group_to_indices.items()
            },
            'category_label': torch.tensor(0, dtype=torch.long),  # Default category label
            'category_type': torch.tensor(0, dtype=torch.long),  # Default category type
            'image_name': 'default'  # Placeholder for image name
        }
    
    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]
            logger.info(f"Loading item {idx}: {row['image_name']}")
            logger.info(f"Image path: {row['cropped_image_path']}")
            logger.info(f"Heatmap path: {row['heatmaps_path']}")

            # Validate file existence
            if not os.path.exists(row['cropped_image_path']):
                logger.warning(f"Image not found: {row['cropped_image_path']}")
                return self._get_default_item()
            if not os.path.exists(row['heatmaps_path']):
                logger.warning(f"Heatmap not found: {row['heatmaps_path']}")
                return self._get_default_item()

            # Load image and heatmap
            image = self._load_image(row['cropped_image_path'])
            heatmap = self._load_heatmap(row['heatmaps_path'])

            # Create attribute targets
            attribute_targets = {
                group: torch.tensor(
                    row.iloc[indices].values.astype(np.float32)
                ) for group, indices in self.group_to_indices.items()
            }

            # Get category labels
            category_label = torch.tensor(int(row['category_label']), dtype=torch.long)
            category_type = torch.tensor(row['category_type'], dtype=torch.long)

            # Apply transformations
            if self.transform:
                seed = np.random.randint(2147483647)  # Same seed for consistent transforms
                random.seed(seed)
                torch.manual_seed(seed)
                image = self.transform(image)

                random.seed(seed)
                torch.manual_seed(seed)
                heatmap = self.transform_heatmap(heatmap)

            return {
                'image': image,
                'heatmap': heatmap,
                'attribute_targets': attribute_targets,
                'category_label': category_label,
                'category_type': category_type,
                'image_name': row['image_name']
            }

        except Exception as e:
            logger.error(f"Error processing item {idx} (image: {row['image_name']}): {str(e)}")
            return self._get_default_item()

    @lru_cache(maxsize=100)
    def _load_image(self, image_path):
        """Loads an image from the given absolute path."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert('RGB')

            # Ensure size is 224x224
            if image.size != (224, 224):
                image = image.resize((224, 224))

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224), color='gray')  # Return a blank image on error

    @lru_cache(maxsize=100)
    def _load_heatmap(self, heatmap_path):
        """Loads a heatmap from the given absolute path and returns a torch tensor."""
        try:
            if not os.path.exists(heatmap_path):
                raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

            heatmap = np.load(heatmap_path).astype(np.float32)
            logger.info(f"Heatmap shape after loading: {heatmap.shape}")

            # Check for NaN or inf values
            if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
                logger.warning(f"Invalid heatmap values (NaN/inf) at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)

            # Log heatmap statistics
            min_val, max_val = heatmap.min(), heatmap.max()
            dynamic_range = max_val - min_val
            logger.info(f"Heatmap stats - min: {min_val}, max: {max_val}, mean: {heatmap.mean()}, std: {heatmap.std()}")

            # Normalize to [0,1] robustly
            if dynamic_range < 1e-3:  # Heatmap is nearly flat
                logger.warning(f"Small dynamic range detected at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)
            elif max_val > min_val:  # Ensure there's a valid range
                heatmap = (heatmap - min_val) / (dynamic_range + 1e-7)  # Avoid division by zero
                heatmap = np.clip(heatmap, 0, 1)  # Clip to [0, 1] in case of outliers
            else:
                logger.warning(f"Flat heatmap detected at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)

            # Ensure size is 224x224
            if heatmap.shape != (224, 224):
                heatmap = np.array(Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR))

            heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # Convert to [1, 224, 224]

            return heatmap

        except Exception as e:
            logger.error(f"Error loading heatmap {heatmap_path}: {str(e)}")
            return torch.zeros((1, 224, 224), dtype=torch.float32)  # Return blank tensor on error


    def default_transforms(self):
        """Default transforms for images and heatmaps"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def transform_heatmap(self, heatmap):
        """Special transforms for heatmaps"""
        if not isinstance(heatmap, torch.Tensor):  # Convert to tensor if not already
            heatmap = torch.from_numpy(heatmap).float()
        if heatmap.ndim == 2:  # Add channel dimension if missing
            heatmap = heatmap.unsqueeze(0)
        logger.info(f"Heatmap shape after transform: {heatmap.shape}")
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, (transforms.RandomHorizontalFlip,
                                transforms.RandomRotation,
                                transforms.RandomAffine,
                                transforms.RandomResizedCrop)):
                    heatmap = t(heatmap)
        return heatmap

    def get_category_mapping(self):
        """Get mapping of category labels to human-readable names"""
        return dict(zip(
            self.metadata['category_label'].unique(),
            self.metadata['category_name'].unique() if 'category_name' in self.metadata.columns
            else self.metadata['category_label'].unique()
        ))

    def get_attribute_names(self):
        """Get list of attribute names in order"""
        return self.metadata.columns[4:-2].tolist()

    def validate_dataset_files(self):
        """Validate that all files exist and have correct dimensions"""
        sample_size = min(100, len(self.metadata))
        indices = random.sample(range(len(self.metadata)), sample_size)
        
        missing_images = 0
        missing_heatmaps = 0
        
        for idx in indices:
            row = self.metadata.iloc[idx]
            img_path = os.path.join(self.cropped_images_dir, row['image_name'])
            heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(row['image_name'])[0]}.npy")
            
            if not os.path.exists(img_path):
                missing_images += 1
            if not os.path.exists(heatmap_path):
                missing_heatmaps += 1

        if missing_images > 0 or missing_heatmaps > 0:
            logger.warning(f"Found {missing_images} missing images and {missing_heatmaps} missing heatmaps in sample of {sample_size}")
        else:
            logger.info(f"Validated {sample_size} random samples, all files exist")

    def get_stats(self):
        """Get dataset statistics"""
        logger.info("Calculating dataset statistics...")

        stats = {
            'num_samples': len(self.metadata),
            'num_attributes': len(self.attribute_groups),
            'num_categories': len(self.metadata['category_label'].unique()),
            'attribute_distributions': {
                group: {
                    attr: self.metadata[attr].value_counts().to_dict()
                    for attr in self.attribute_groups[group]
                } for group in self.attribute_groups
            },
            'category_distribution': self.metadata['category_label'].value_counts().to_dict(),
            'type_distribution': self.metadata['category_type'].value_counts().to_dict()
        }

        return stats

if __name__ == "__main__":
    # Define attribute groups
    ATTRIBUTE_GROUPS = {
        'color_print': ['red', 'pink', 'floral', 'striped', 'stripe', 'print', 'printed', 'graphic', 'love', 'summer'],
        'neckline': ['v-neck', 'hooded', 'collar', 'sleeveless', 'strapless', 'racerback', 'muscle'],
        'silhouette_fit': ['slim', 'boxy', 'fit', 'skinny', 'shift', 'bodycon', 'maxi', 'mini', 'midi', 'a-line'],
        'style_construction': ['crochet', 'knit', 'woven', 'lace', 'denim', 'cotton', 'chiffon', 'mesh'],
        'details': ['pleated', 'pocket', 'button', 'drawstring', 'trim', 'hem', 'capri', 'sleeve', 'flare', 'skater', 'sheath', 'shirt', 'pencil', 'classic', 'crop']
    }

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Path configuration
    metadata_path = "zsehran/metadata_updated_train.csv"  # Update with your path
    cropped_images_dir = "data/processed/cropped_images"
    heatmaps_dir = "data/processed/heatmaps"

    # Initialize the dataset
    dataset = FashionDataset(
        metadata_path=metadata_path,
        cropped_images_dir=cropped_images_dir,
        heatmaps_dir=heatmaps_dir,
        attribute_groups=ATTRIBUTE_GROUPS,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True
    )

    # Print dataset statistics
    stats = dataset.get_stats()
    print("\nDataset Statistics:")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Number of attributes: {stats['num_attributes']}")
    print(f"Number of categories: {stats['num_categories']}")
    print(f"Category distribution: {stats['category_distribution']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print("\nAttribute distributions:")
    for group, attr_dist in stats['attribute_distributions'].items():
        print(f"\n{group}:")
        for attr, dist in attr_dist.items():
            print(f"  {attr}: {dist}")

    # Create a DataLoader with num_workers=0
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Iterate through the DataLoader
    print("\nTesting DataLoader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Images shape: {batch['image'].shape}")               # [B, 3, 224, 224]
        print(f"Heatmaps shape: {batch['heatmap'].shape}")           # [B, 1, 224, 224]
        print(f"Category labels: {batch['category_label']}")         # [B]
        print(f"Category types: {batch['category_type']}")           # [B]
        
        # Print attribute targets for each group
        for group, targets in batch['attribute_targets'].items():
            print(f"{group} attributes shape: {targets.shape}")      # [B, num_attributes_in_group]
        
        # Break after a few batches for demonstration
        if batch_idx == 2:
            break

    print("\nDataset script test completed successfully!")