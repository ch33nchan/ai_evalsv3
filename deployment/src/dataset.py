import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from typing import List, Tuple
from tqdm import tqdm

class SlopDatasetGenerator:
    def __init__(self, cache_dir: str = './data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, train_samples: int = 500, val_samples: int = 100) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        total_samples = train_samples + val_samples
        num_per_class = total_samples // 2
        
        base_images = self._generate_base_images(num_per_class)
        slop_images = self._generate_slop_images(num_per_class)
        
        base_labeled = [(str(path), 0) for path in base_images]
        slop_labeled = [(str(path), 1) for path in slop_images]
        
        all_data = base_labeled + slop_labeled
        np.random.shuffle(all_data)
        
        train_data = all_data[:train_samples]
        val_data = all_data[train_samples:train_samples + val_samples]
        
        return train_data, val_data
    
    def _generate_base_images(self, num_samples: int) -> List[Path]:
        paths = []
        for i in tqdm(range(num_samples), desc="Generating base images"):
            path = self.cache_dir / f"base_{i}.jpg"
            if not path.exists():
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                img.save(path, quality=95)
            paths.append(path)
        return paths
    
    def _generate_slop_images(self, num_samples: int) -> List[Path]:
        paths = []
        for i in tqdm(range(num_samples), desc="Generating slop images"):
            path = self.cache_dir / f"slop_{i}.jpg"
            if not path.exists():
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                
                noise = np.random.normal(0, 15, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.SMOOTH_MORE)
                img = img.filter(ImageFilter.SMOOTH)
                
                img.save(path, quality=75)
            paths.append(path)
        return paths
