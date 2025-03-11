import logging
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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
    """
    Custom Dataset class for loading fashion items with:
    - Cropped images (224x224 RGB)
    - Landmark heatmaps (224x22 4 single-channel)
    - Metadata (50 attributes + category information)
    """

    def __init__(self, metadata_path, cropped_images_dir, heatmaps_dir, transform=None,
                 use_cache=True, cache_size=100, validate_files=True):

        """
        Args:
            metadata_path (str): Path to metadata CSV file
            cropped_images_dir (str): Root directory for cropped images
            heatmaps_dir (str): Root directory for heatmap numpy files
            transform (callable, optional): Optional transform to be applied
            use_cache (bool): Whether to cache loaded images and heatmaps in memory
            cache_size (int): Size of LRU cache for images and heatmaps
            validate_files (bool): Whether to validate existence of files during initialization
        """
        super().__init__()
        logger.info(f"Initializing FashionDataset with metadata: {metadata_path}")
        print((f"Loading metadata from: {metadata_path}"))
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

        # if exclude_categories is not None:
        #     logger.info(f"Excluding categories: {exclude_categories}")
        #     self.metadata = self.metadata[~self.metadata['category_label'].isin(exclude_categories)]
        #     logger.info(f"Filtered metadata contains {len(self.metadata)} entries")

        # Store paths
        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir

        # Define transforms
        self.transform = transform or self.default_transforms()
        logger.info(f"Using transforms: {self.transform}")

        # Extract attribute columns (first 50 attributes in CSV)
        self.attribute_columns = self.metadata.columns[4:-2].tolist()  # Skip first 4 and last 2 columns
        logger.info(f"Extracted {len(self.attribute_columns)} attribute columns")

        # --- REMAP CATEGORY LABELS ---
        # Original unique categories
        original_categories = sorted(self.metadata['category_label'].unique())
        logger.info(f"Original unique categories: {original_categories}")

        # Ensure valid categories (removing missing ones)
        valid_categories = [c for c in range(1, 51) if c not in [38, 45, 49, 50]]  # Categories 1-50 except missing ones
        logger.info(f"Valid categories after filtering: {valid_categories}")

        # Store original category labels before mapping
        self.metadata['original_category_label'] = self.metadata['category_label'].copy()

        # Create a mapping from original labels to contiguous labels
        self.category_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_categories))}

        logger.info(f"Category mapping: {self.category_mapping}")

        # Apply mapping to category labels
        self.metadata = self.metadata[self.metadata['category_label'].isin(valid_categories)]  # Drop missing categories
        self.metadata['category_label'] = self.metadata['category_label'].map(self.category_mapping)

        assert self.metadata['category_label'].min() == 0, "Error: Category labels should start from 0"
        assert self.metadata['category_label'].max() == 45, "Error: Category labels should range from 0 to 45"


        unique_labels = sorted(self.metadata['category_label'].unique().tolist())
        print("Final Unique Categories in Dataset:", unique_labels)
        assert 45 in unique_labels, "Error: Category 45 is missing from the dataset!"




         # Verify the labels are now 0-indexed
        min_label = self.metadata['category_label'].min()
        max_label = self.metadata['category_label'].max()
        logger.info(f"After remapping: category_label min={min_label}, max={max_label}")

        # Convert category types to 0-based index (if needed)
        self.metadata['category_type'] = self.metadata['category_type'] - 1  # Adjust from [1-3] to [0-2]


        # Debugging: Verify labels are now 0-indexed
        logger.info(f"Final category labels after remapping: {sorted(self.metadata['category_label'].unique())}")
        print(f"Final Category Labels: {sorted(self.metadata['category_label'].unique())}")  # Should show [0, 1, ..., 45]

        # Category information
        self.category_labels = self.metadata['category_label']
        self.category_types = self.metadata['category_type']

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
            self.category_names = dict(zip(self.metadata['category_label'].unique(),
                                        self.metadata['category_label'].unique()))

        # Caching
        self.cache_size = cache_size  # store cache_size for later use
        self.use_cache = use_cache
        if use_cache:
            logger.info(f"Enabling LRU cache with size {cache_size}")
            self.load_image = lru_cache(maxsize=cache_size)(self._load_image)
            self.load_heatmap = lru_cache(maxsize=cache_size)(self._load_heatmap)
        else:
            self.load_image = self._load_image
            self.load_heatmap = self._load_heatmap


        # Validate files if requested
        if validate_files:
            self.validate_dataset_files()

    def validate_dataset_files(self):
        """Validate that all files exist and have correct dimensions"""
        logger.info("Validating dataset files...")

        missing_images = 0
        missing_heatmaps = 0

        # Check a subset of files when dataset is large
        sample_size = min(100, len(self.metadata))
        indices = random.sample(range(len(self.metadata)), sample_size)

        for idx in indices:
            row = self.metadata.iloc[idx]

            # Check image file
            img_path = os.path.join(self.cropped_images_dir, row['image_name'])


            if not os.path.exists(img_path):
                missing_images += 1
                logger.warning(f"Image not found: {img_path}")

            # Check heatmap file
            heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(row['image_name'])[0]}.npy")
            if not os.path.exists(heatmap_path):
                missing_heatmaps += 1
                logger.warning(f"Heatmap not found: {heatmap_path}")

        if missing_images > 0 or missing_heatmaps > 0:
            logger.warning(f"Found {missing_images} missing images and {missing_heatmaps} missing heatmaps in sample of {sample_size}")
        else:
            logger.info(f"Validated {sample_size} random samples, all files exist")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            # Get metadata row
            row = self.metadata.iloc[idx]

            # Load cropped image
            image = self.load_image(row['image_name'])

            # Load heatmap
            heatmap = self.load_heatmap(row['image_name'])

            # Get attributes (convert to float tensor)
            attributes = torch.tensor(row[self.attribute_columns].values.astype(np.float32))

            # Get category information
            category_label = torch.tensor(int(row['category_label']), dtype=torch.long)
            assert category_label >= 0, f"Error: Found category -1 in dataset at index {idx}!"
            category_type = torch.tensor(row['category_type'], dtype=torch.long)

            # Apply transforms
            if self.transform:
                # Use same random state for consistent transforms between image and heatmap
                seed = np.random.randint(2147483647)

                # Convert image to tensor before applying transform
                if not isinstance(image, torch.Tensor):
                    image = transforms.ToTensor()(image)  #  Convert PIL image to tensor first


                # Check if `ToTensor()` exists in `self.transform`
                if not any(isinstance(t, transforms.ToTensor) for t in self.transform.transforms):
                    image = self.transform(image)

                # Apply same random seed for reproducibility
                random.seed(seed)
                torch.manual_seed(seed)

                # Apply same spatial transforms to heatmap
                random.seed(seed)
                torch.manual_seed(seed)
                heatmap = self.transform_heatmap(heatmap)

            return {
                'image': image,
                'heatmap': heatmap,
                'attributes': attributes,
                'category_label': category_label,
                'category_type': category_type,
                'image_name': row['image_name']  # For debugging/visualization
            }



        except Exception as e:
            logger.error(f"Error in __getitem__: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Error processing item {idx} ({row['image_name'] if 'row' in locals() else 'unknown'}): {str(e)}")
            # Return a default item or re-raise
            if idx > 0:  # Try to return a different item if possible
                return self.__getitem__(0)
            else:
                raise

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the cached function attributes
        if 'load_image' in state:
            del state['load_image']
        if 'load_heatmap' in state:
            del state['load_heatmap']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the cached functions based on the stored flags
        if self.use_cache:
            self.load_image = lru_cache(maxsize=self.cache_size)(self._load_image)
            self.load_heatmap = lru_cache(maxsize=self.cache_size)(self._load_heatmap)
        else:
            self.load_image = self._load_image
            self.load_heatmap = self._load_heatmap


    def _load_image(self, image_name):
        """Helper function to load image with error handling"""
        img_path = os.path.join(self.cropped_images_dir, image_name)
        try:
            if not os.path.exists(img_path):
                logger.error(f"Image not found: {img_path}")
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = Image.open(img_path).convert('RGB')
            # Verify dimensions
            if image.size != (224, 224):
                logger.warning(f"Image {img_path} has unexpected dimensions {image.size}, resizing to (224, 224)")
                image = image.resize((224, 224))

            return image

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='gray')

    def _load_heatmap(self, image_name):
        """Helper function to load heatmap with error handling"""
        heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(image_name)[0]}.npy")
        try:
            if not os.path.exists(heatmap_path):
                logger.error(f"Heatmap not found: {heatmap_path}")
                raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

            heatmap = np.load(heatmap_path)
            # Verify dimensions
            if heatmap.shape != (224, 224):
                logger.warning(f"Heatmap {heatmap_path} has unexpected dimensions {heatmap.shape}, resizing to (224, 224)")
                # Resize using simple interpolation
                heatmap = np.array(Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR))

            return heatmap

        except Exception as e:
            logger.error(f"Error loading heatmap {heatmap_path}: {str(e)}")
            # Return a blank heatmap as fallback
            return np.zeros((224, 224), dtype=np.float32)

    def default_transforms(self):
        """Default transforms for images and heatmaps"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])

    def transform_heatmap(self, heatmap):
        """Special transforms for heatmaps"""
        # Convert to tensor and add channel dimension
        heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)  # [1, H, W]

        # Apply same spatial transforms as image if needed
        if self.transform:
            # Only apply spatial transforms, not normalization
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, (transforms.RandomHorizontalFlip,
                                      transforms.RandomRotation,
                                      transforms.RandomAffine,
                                      transforms.RandomResizedCrop)):
                        heatmap = t(heatmap)
        return heatmap

    def get_category_mapping(self):
        """Get mapping of category labels to human-readable names"""
        return self.category_names

    def get_attribute_names(self):
        """Get list of attribute names in order"""
        return self.attribute_columns

    def get_stats(self):
        """Get dataset statistics"""
        logger.info("Calculating dataset statistics...")

        stats = {
            'num_samples': len(self.metadata),
            'num_attributes': len(self.attribute_columns),
            'num_categories': len(self.category_labels.unique()),
            'attribute_distributions': {attr: self.metadata[attr].value_counts().to_dict()
                                        for attr in self.attribute_columns[:5]},  # First 5 for brevity
            'category_distribution': self.metadata['category_label'].value_counts().to_dict(),
            'type_distribution': self.metadata['category_type'].value_counts().to_dict()
        }

        return stats

# Example usage
if __name__ == "__main__":
    # Path configuration
    metadata_path = "data/processed/metadata_updated.csv"
    cropped_images_dir =  "data/processed/cropped_images1"
    heatmaps_dir = "data/processed/heatmaps1"

    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    logger.info("Creating dataset instance...")

    # Create dataset
    dataset = FashionDataset(
        metadata_path=metadata_path,
        cropped_images_dir=cropped_images_dir,
        heatmaps_dir=heatmaps_dir,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True
    )

    logger.info(f"Dataset initialized with {len(dataset)} samples")

    # Get dataset statistics
    stats = dataset.get_stats()
    logger.info(f"Dataset statistics: {stats}")

    # Example batch
    logger.info("Fetching sample batch...")
    sample = dataset[0]

    logger.info("Sample keys: %s", sample.keys())
    logger.info("Image shape: %s", sample['image'].shape)
    logger.info("Heatmap shape: %s", sample['heatmap'].shape)
    logger.info("Attributes shape: %s", sample['attributes'].shape)
    logger.info("Category label: %s", sample['category_label'])
    logger.info("Category type: %s", sample['category_type'])

    # Test DataLoader
    logger.info("Testing DataLoader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, batch in enumerate(loader):
        if i == 0:
            logger.info(f"First batch shapes: Image {batch['image'].shape}, Heatmap {batch['heatmap'].shape}")

        if i >= 2:  # Only test a few batches
            break

    logger.info("DataLoader test completed successfully")