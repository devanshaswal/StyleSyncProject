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

# Set up logging - reduce file logging in Colab
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console logging for Colab
    ]
)
logger = logging.getLogger('FashionDataset')


class FashionDataset(Dataset):
    """
    Memory-optimized Dataset class for loading fashion items with:
    - Cropped images (224x224 RGB)
    - Landmark heatmaps (224x224 single-channel)
    - Metadata (attributes + category information)
    """
    
    def __init__(self, metadata_path, cropped_images_dir, heatmaps_dir, transform=None, 
                 use_cache=True, cache_size=50, validate_files=False):
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
        
        # Store paths
        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir
        
        # Define transforms
        self.transform = transform or self.default_transforms()
        
        # Memory optimization: First load a sample to determine columns
        sample_df = pd.read_csv(metadata_path, nrows=5)
        self.attribute_columns = sample_df.columns[4:-2].tolist()  # Skip first 4 and last 2 columns
        
        # Memory optimization: Only load necessary columns
        try:
            columns_to_load = ['image_name', 'category_label', 'category_type'] + self.attribute_columns
            self.metadata = pd.read_csv(metadata_path, usecols=columns_to_load)
        except:
            # If column specification fails, load then filter
            self.metadata = pd.read_csv(metadata_path)
            keep_columns = ['image_name', 'category_label', 'category_type'] + self.attribute_columns
            self.metadata = self.metadata[keep_columns]
        
        logger.info(f"Loaded metadata with {len(self.metadata)} entries and {len(self.attribute_columns)} attributes")
        
        # Category information
        self.category_labels = self.metadata['category_label']
        self.category_types = self.metadata['category_type']
        
        # Create simple category name mapping
        unique_categories = self.metadata['category_label'].unique()
        self.category_names = dict(zip(unique_categories, unique_categories))
        
        # Caching - properly implemented
        self.use_cache = use_cache
        if use_cache:
            logger.info(f"Enabling LRU cache with size {cache_size}")
            self.load_image = lru_cache(maxsize=cache_size)(self._load_image)
            self.load_heatmap = lru_cache(maxsize=cache_size)(self._load_heatmap)
        else:
            self.load_image = self._load_image
            self.load_heatmap = self._load_heatmap
        
        # Validate files if requested (limited sample size)
        if validate_files:
            self.validate_dataset_files(max_samples=20)

    def validate_dataset_files(self, max_samples=20):
        """Validate that files exist with reduced memory footprint"""
        logger.info("Validating a subset of dataset files...")
        
        missing_images = 0
        missing_heatmaps = 0
        
        # Check a small subset of files
        sample_size = min(max_samples, len(self.metadata))
        indices = random.sample(range(len(self.metadata)), sample_size)
        
        for idx in indices:
            row = self.metadata.iloc[idx]
            
            # Check image file
            img_path = os.path.join(self.cropped_images_dir, row['image_name'])
            if not os.path.exists(img_path):
                missing_images += 1
            
            # Check heatmap file
            heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(row['image_name'])[0]}.npy")
            if not os.path.exists(heatmap_path):
                missing_heatmaps += 1
        
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
            logger.error(f"Error processing item {idx}: {str(e)}")
            # Return a default item or re-raise
            if idx > 0:  # Try to return a different item if possible
                return self.__getitem__(0)
            else:
                raise
    
    def _load_image(self, image_name):
        """Memory-efficient image loading"""
        img_path = os.path.join(self.cropped_images_dir, image_name)
        try:
            if not os.path.exists(img_path):
                return Image.new('RGB', (224, 224), color='gray')
            
            image = Image.open(img_path).convert('RGB')
            # Verify dimensions
            if image.size != (224, 224):
                image = image.resize((224, 224))
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return Image.new('RGB', (224, 224), color='gray')
    
    def _load_heatmap(self, image_name):
        """Memory-efficient heatmap loading"""
        heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(image_name)[0]}.npy")
        try:
            if not os.path.exists(heatmap_path):
                return np.zeros((224, 224), dtype=np.float32)
            
            heatmap = np.load(heatmap_path)
            # Verify dimensions
            if heatmap.shape != (224, 224):
                heatmap = np.array(Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR))
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error loading heatmap {heatmap_path}: {str(e)}")
            return np.zeros((224, 224), dtype=np.float32)
    
    def default_transforms(self):
        """Default transforms for images and heatmaps"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def transform_heatmap(self, heatmap):
        """Special transforms for heatmaps"""
        # Convert to tensor and add channel dimension
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]
        
        # Apply only spatial transforms from the transform pipeline
        if self.transform and isinstance(self.transform, transforms.Compose):
            for t in self.transform.transforms:
                if isinstance(t, (transforms.RandomHorizontalFlip,
                                transforms.RandomRotation,
                                transforms.RandomAffine,
                                transforms.RandomResizedCrop)):
                    heatmap = t(heatmap)
        return heatmap

    def get_category_mapping(self):
        """Get mapping of category labels to names"""
        return self.category_names

    def get_attribute_names(self):
        """Get list of attribute names in order"""
        return self.attribute_columns
    
    def get_stats(self, sample_size=100):
        """Get dataset statistics with memory optimization"""
        logger.info("Calculating dataset statistics (sampled)...")
        
        # Sample the dataset for statistics to save memory
        if len(self.metadata) > sample_size:
            sample_df = self.metadata.sample(sample_size)
        else:
            sample_df = self.metadata
            
        stats = {
            'num_samples': len(self.metadata),
            'num_attributes': len(self.attribute_columns),
            'num_categories': len(self.category_labels.unique()),
            'category_distribution': self.metadata['category_label'].value_counts().head(10).to_dict(),
        }
        
        return stats


# Usage example - make conditional to avoid automatic execution
if __name__ == "__main__":
    # Path configuration
    metadata_path = "/content/drive/MyDrive/stylesync/metadata_updated.csv"
    cropped_images_dir = '/content/cropped_images/cropped_images'
    heatmaps_dir = '/content/heatmaps/heatmaps'
    
    # Create dataset with memory-optimized settings
    dataset = FashionDataset(
        metadata_path=metadata_path,
        cropped_images_dir=cropped_images_dir,
        heatmaps_dir=heatmaps_dir,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]),
        use_cache=True,
        cache_size=50,  # Reduced cache size
        validate_files=False  # Skip validation on init
    )
    
    print(f"Dataset initialized with {len(dataset)} samples")
    
    # Test a single sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Heatmap shape: {sample['heatmap'].shape}")
    
    # Test DataLoader with small batch size
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Test first batch only
    batch = next(iter(loader))
    print(f"Batch shapes: Image {batch['image'].shape}, Heatmap {batch['heatmap'].shape}")