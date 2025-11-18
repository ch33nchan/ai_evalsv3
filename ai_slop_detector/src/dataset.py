import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from typing import List, Tuple

class SlopDatasetGenerator:
    def __init__(self, cache_dir: str = './data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_synthetic_images(self, num_samples: int = 100) -> List[str]:
        synthetic_paths = []
        
        for idx in range(num_samples):
            save_path = self.cache_dir / f"base_{idx}.jpg"
            
            if save_path.exists():
                synthetic_paths.append(str(save_path))
                continue
            
            height, width = np.random.randint(256, 512), np.random.randint(256, 512)
            
            base_color = tuple(np.random.randint(30, 230, 3).tolist())
            img = Image.new('RGB', (width, height), base_color)
            draw = ImageDraw.Draw(img)
            
            num_shapes = np.random.randint(5, 12)
            for _ in range(num_shapes):
                shape_type = np.random.choice(['circle', 'rectangle', 'line'])
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                if shape_type == 'circle':
                    x, y = np.random.randint(0, width), np.random.randint(0, height)
                    radius = np.random.randint(30, min(width, height) // 3)
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                elif shape_type == 'rectangle':
                    x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                    x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                    x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                    thickness = np.random.randint(3, 15)
                    draw.line([x1, y1, x2, y2], fill=color, width=thickness)
            
            np_img = np.array(img)
            high_freq_noise = np.random.normal(0, 25, np_img.shape)
            np_img = np.clip(np_img + high_freq_noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(np_img)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            img.save(save_path, quality=98)
            synthetic_paths.append(str(save_path))
            
            if (idx + 1) % 20 == 0:
                print(f"Generated base image {idx + 1}/{num_samples}")
        
        return synthetic_paths
    
    def generate_slop_variants(self, base_images: List[str], num_samples: int = 100) -> List[str]:
        slop_paths = []
        
        for idx in range(num_samples):
            base_path = base_images[idx % len(base_images)]
            save_path = self.cache_dir / f"slop_{idx}.jpg"
            
            if save_path.exists():
                slop_paths.append(str(save_path))
                continue
            
            img = Image.open(base_path).convert('RGB')
            np_img = np.array(img)
            
            noise_level = np.random.uniform(0.05, 0.15)
            noise = np.random.normal(0, noise_level * 255, np_img.shape)
            slop_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            
            slop_img = Image.fromarray(slop_img)
            slop_img = slop_img.filter(ImageFilter.SMOOTH_MORE)
            slop_img = slop_img.filter(ImageFilter.GaussianBlur(radius=2.5))
            slop_img = slop_img.filter(ImageFilter.SMOOTH)
            
            enhancer = ImageEnhance.Sharpness(slop_img)
            slop_img = enhancer.enhance(0.5)
            
            slop_img.save(save_path, quality=65)
            slop_paths.append(str(save_path))
            
            if (idx + 1) % 20 == 0:
                print(f"Generated slop image {idx + 1}/{num_samples}")
        
        return slop_paths
    
    def create_dataset(self, train_samples: int = 500, val_samples: int = 100) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        total_samples = train_samples + val_samples
        samples_per_class = total_samples // 2
        
        print(f"Generating {samples_per_class} base images...")
        base_images = self.generate_synthetic_images(samples_per_class)
        
        print(f"Generating {samples_per_class} slop variants...")
        slop_images = self.generate_slop_variants(base_images, samples_per_class)
        
        real_labeled = [(path, 0) for path in base_images]
        slop_labeled = [(path, 1) for path in slop_images]
        
        all_data = real_labeled + slop_labeled
        np.random.shuffle(all_data)
        
        train_data = all_data[:train_samples]
        val_data = all_data[train_samples:train_samples + val_samples]
        
        return train_data, val_data
