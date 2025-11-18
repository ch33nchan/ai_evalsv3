import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from scipy import fftpack
from scipy.stats import entropy
import cv2
import imagehash
from typing import Dict, Tuple

class MultiModalFeatureExtractor:
    def __init__(self, image_size: int = 224, device: str = 'cpu'):
        self.image_size = image_size
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.perceptual_model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.perceptual_model.eval()
        
    def extract_frequency_features(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq = magnitude[center_h-20:center_h+20, center_w-20:center_w+20].mean()
        high_freq = (magnitude[:20, :].mean() + magnitude[-20:, :].mean() + 
                     magnitude[:, :20].mean() + magnitude[:, -20:].mean()) / 4
        
        freq_ratio = low_freq / (high_freq + 1e-8)
        freq_entropy = entropy(magnitude.flatten() + 1e-8)
        
        return np.array([low_freq, high_freq, freq_ratio, freq_entropy])
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        glcm_contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / edges.size
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_std = gradient_mag.std()
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        texture_entropy = entropy(hist + 1e-8)
        
        return np.array([glcm_contrast, edge_density, gradient_std, texture_entropy])
    
    def extract_perceptual_features(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.perceptual_model(img_tensor)
            return features.squeeze().cpu().numpy()
    
    def extract_semantic_artifacts(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        block_size = 16
        smoothness_scores = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                if len(block.shape) == 3:
                    block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
                smoothness = np.std(block)
                smoothness_scores.append(smoothness)
        
        if not smoothness_scores:
            return np.array([0.0, 0.0, 0.0])
            
        smoothness_mean = np.mean(smoothness_scores)
        smoothness_std = np.std(smoothness_scores)
        smoothness_ratio = smoothness_std / (smoothness_mean + 1e-8)
        
        return np.array([smoothness_mean, smoothness_std, smoothness_ratio])
    
    def extract_all_features(self, image_path: str) -> Dict[str, np.ndarray]:
        pil_image = Image.open(image_path).convert('RGB')
        np_image = np.array(pil_image)
        
        features = {
            'frequency': self.extract_frequency_features(np_image),
            'texture': self.extract_texture_features(np_image),
            'perceptual': self.extract_perceptual_features(pil_image),
            'semantic': self.extract_semantic_artifacts(np_image)
        }
        
        return features
    
    def features_to_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        perceptual_compressed = features['perceptual'][:64]
        return np.concatenate([
            features['frequency'],
            features['texture'],
            perceptual_compressed,
            features['semantic']
        ])
