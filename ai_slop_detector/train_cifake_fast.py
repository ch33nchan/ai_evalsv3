import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

from src.feature_extractor import MultiModalFeatureExtractor
from src.rl_agent import RLAgent
from src.reward_model import RewardModel

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

with open('data/cifake_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/cifake_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

device = 'cuda:0'

cache_file = Path('data/cifake_features_cache.pkl')

if cache_file.exists():
    print("Loading cached features...")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    train_features = cache['train']
    val_features = cache['val']
else:
    print("Extracting features (one-time operation)...")
    extractor = MultiModalFeatureExtractor(224, device)
    
    train_features = []
    for img_path, label in tqdm(train_data, desc="Train features"):
        try:
            features = extractor.extract_all_features(img_path)
            state = extractor.features_to_vector(features)
            train_features.append((state, label))
        except:
            continue
    
    val_features = []
    for img_path, label in tqdm(val_data, desc="Val features"):
        try:
            features = extractor.extract_all_features(img_path)
            state = extractor.features_to_vector(features)
            val_features.append((state, label))
        except:
            continue
    
    with open(cache_file, 'wb') as f:
        pickle.dump({'train': train_features, 'val': val_features}, f)
    print(f"Cached {len(train_features)} train, {len(val_features)} val features")

agent = RLAgent(75, 128, 10, 0.0003, 0.99, device)
reward_model = RewardModel()

best_val_acc = 0.0
history = {'train_acc': [], 'val_acc': []}

pbar = tqdm(range(150), desc="Training")

for episode in pbar:
    np.random.shuffle(train_features)
    
    correct = 0
    for state, true_label in train_features:
        action = agent.select_action(state)
        predicted_label = 1 if action >= 5 else 0
        
        if predicted_label == true_label:
            correct += 1
            reward = 1.0
        else:
            reward = -1.0
        
        agent.store_transition(state, action, reward, state, True)
        agent.train_step(32)
    
    train_acc = correct / len(train_features)
    
    val_correct = 0
    for state, true_label in val_features:
        action = agent.select_action(state)
        predicted_label = 1 if action >= 5 else 0
        if predicted_label == true_label:
            val_correct += 1
    
    val_acc = val_correct / len(val_features)
    
    if episode % 10 == 0:
        agent.update_target_network()
    
    agent.decay_epsilon()
    
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        agent.save('models/best_model.pt')
        with open('data/val_data.pkl', 'wb') as f:
            pickle.dump([(f"cached_{i}", label) for i, (_, label) in enumerate(val_features)], f)
    
    pbar.set_postfix({
        'train_acc': f'{train_acc:.3f}',
        'val_acc': f'{val_acc:.3f}',
        'best': f'{best_val_acc:.3f}',
        'eps': f'{agent.epsilon:.3f}'
    })

print(f"Training complete. Best: {best_val_acc:.3f}")
