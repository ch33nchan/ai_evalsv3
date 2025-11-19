import yaml
import pickle
from src.train import Trainer

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

with open('data/cifake_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('data/cifake_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

print(f"Training on {len(train_data)} samples, validating on {len(val_data)}")

config['training']['num_episodes'] = 150

trainer = Trainer(config, device='cuda')
history = trainer.train(train_data, val_data, config['training']['num_episodes'])

print(f"Best validation accuracy: {max(history['val_acc']):.3f}")

with open('data/val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)
