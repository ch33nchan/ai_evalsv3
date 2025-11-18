import numpy as np
from typing import Dict

class RewardModel:
    def __init__(self):
        self.feature_importance = {
            'frequency': 0.3,
            'texture': 0.25,
            'perceptual': 0.3,
            'semantic': 0.15
        }
        
    def compute_reward(self, features: Dict[str, np.ndarray], 
                      action_weights: np.ndarray, 
                      true_label: int,
                      predicted_label: int) -> float:
        
        correctness_reward = 1.0 if predicted_label == true_label else -1.0
        
        weighted_features = []
        for feature_name, feature_vec in features.items():
            importance = self.feature_importance[feature_name]
            weighted_features.append(np.mean(feature_vec) * importance)
        
        feature_quality = np.mean(weighted_features)
        
        confidence_penalty = -0.1 * np.std(action_weights)
        
        total_reward = correctness_reward + 0.2 * feature_quality + confidence_penalty
        
        return total_reward
    
    def update_importance(self, feature_name: str, delta: float):
        if feature_name in self.feature_importance:
            self.feature_importance[feature_name] = np.clip(
                self.feature_importance[feature_name] + delta, 0.0, 1.0
            )
            total = sum(self.feature_importance.values())
            for key in self.feature_importance:
                self.feature_importance[key] /= total
