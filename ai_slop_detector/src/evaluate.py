import numpy as np
from typing import List, Tuple, Dict

from .feature_extractor import MultiModalFeatureExtractor
from .rl_agent import RLAgent

class Evaluator:
    def __init__(self, config: Dict, model_path: str, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        self.feature_extractor = MultiModalFeatureExtractor(
            image_size=config['data']['image_size'],
            device=device
        )
        
        feature_dim = 4 + 4 + 64 + 3
        
        self.agent = RLAgent(
            state_dim=feature_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_actions=config['model']['num_actions'],
            learning_rate=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            device=device
        )
        
        self.agent.load(model_path)
        self.agent.epsilon = 0.0
        
    def predict(self, image_path: str) -> Tuple[int, float]:
        features = self.feature_extractor.extract_all_features(image_path)
        state = self.feature_extractor.features_to_vector(features)
        
        action = self.agent.select_action(state, explore=False)
        predicted_label = 1 if action >= self.config['model']['num_actions'] // 2 else 0
        confidence = action / self.config['model']['num_actions']
        
        return predicted_label, confidence
    
    def evaluate_dataset(self, dataset: List[Tuple[str, int]]) -> Dict:
        correct = 0
        predictions = []
        
        for image_path, true_label in dataset:
            try:
                pred_label, confidence = self.predict(image_path)
                predictions.append((pred_label, true_label, confidence))
                
                if pred_label == true_label:
                    correct += 1
            except Exception as e:
                continue
        
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'total': len(dataset),
            'correct': correct
        }
