import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path

from .feature_extractor import MultiModalFeatureExtractor
from .rl_agent import RLAgent
from .reward_model import RewardModel

class Trainer:
    def __init__(self, config: Dict, device: str = 'cpu'):
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
        
        self.reward_model = RewardModel()
        
    def train_episode(self, dataset: List[Tuple[str, int]]) -> Tuple[float, float, float]:
        episode_reward = 0.0
        episode_loss = 0.0
        correct = 0
        
        for image_path, true_label in dataset:
            try:
                features = self.feature_extractor.extract_all_features(image_path)
                state = self.feature_extractor.features_to_vector(features)
                
                action = self.agent.select_action(state)
                predicted_label = 1 if action >= self.config['model']['num_actions'] // 2 else 0
                
                action_weights = np.zeros(self.config['model']['num_actions'])
                action_weights[action] = 1.0
                
                reward = self.reward_model.compute_reward(
                    features, action_weights, true_label, predicted_label
                )
                
                next_state = state
                done = True
                
                self.agent.store_transition(state, action, reward, next_state, done)
                
                loss = self.agent.train_step(self.config['rl']['batch_size'])
                
                episode_reward += reward
                episode_loss += loss
                
                if predicted_label == true_label:
                    correct += 1
                    
            except Exception as e:
                continue
        
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
        return episode_reward, episode_loss, accuracy
    
    def train(self, train_data: List[Tuple[str, int]], 
              val_data: List[Tuple[str, int]],
              num_episodes: int) -> Dict:
        
        best_val_acc = 0.0
        history = {'train_reward': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        pbar = tqdm(range(num_episodes), desc="Training Progress")
        
        for episode in pbar:
            np.random.shuffle(train_data)
            
            train_reward, train_loss, train_acc = self.train_episode(train_data)
            
            _, _, val_acc = self.evaluate(val_data)
            
            if episode % self.config['rl']['target_update'] == 0:
                self.agent.update_target_network()
            
            self.agent.decay_epsilon()
            
            history['train_reward'].append(train_reward)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.agent.save('models/best_model.pt')
            
            pbar.set_postfix({
                'train_acc': f'{train_acc:.3f}',
                'val_acc': f'{val_acc:.3f}',
                'best_val': f'{best_val_acc:.3f}',
                'eps': f'{self.agent.epsilon:.3f}'
            })
        
        return history
    
    def evaluate(self, dataset: List[Tuple[str, int]]) -> Tuple[float, float, float]:
        total_reward = 0.0
        correct = 0
        
        for image_path, true_label in dataset:
            try:
                features = self.feature_extractor.extract_all_features(image_path)
                state = self.feature_extractor.features_to_vector(features)
                
                action = self.agent.select_action(state)
                predicted_label = 1 if action >= self.config['model']['num_actions'] // 2 else 0
                
                if predicted_label == true_label:
                    correct += 1
                    
            except Exception as e:
                continue
        
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
        return 0.0, 0.0, accuracy
