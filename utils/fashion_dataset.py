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
    - Landmark heatmaps (224x224 single-channel)
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
        
        # Store paths
        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir
        
        # Define transforms
        self.transform = transform or self.default_transforms()
        logger.info(f"Using transforms: {self.transform}")
        
        # Extract attribute columns (first 50 attributes in CSV)
        self.attribute_columns = self.metadata.columns[4:-2].tolist()  # Skip first 4 and last 2 columns
        logger.info(f"Extracted {len(self.attribute_columns)} attribute columns")
        
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
            category_label = torch.tensor(row['category_label'], dtype=torch.long)
            category_type = torch.tensor(row['category_type'], dtype=torch.long)
            
            # Apply transforms
            if self.transform:
                # Use same random state for consistent transforms between image and heatmap
                seed = np.random.randint(2147483647)
                
                # Apply transforms to image
                random.seed(seed)
                torch.manual_seed(seed)
                image = self.transform(image)
                
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
            logger.error(f"Error processing item {idx} ({row['image_name'] if 'row' in locals() else 'unknown'}): {str(e)}")
            # Return a default item or re-raise
            if idx > 0:  # Try to return a different item if possible
                return self.__getitem__(0)
            else:
                raise
    
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
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]
        
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
    cropped_images_dir = "data/processed/cropped_images"
    heatmaps_dir = "data/processed/heatmaps"
    
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