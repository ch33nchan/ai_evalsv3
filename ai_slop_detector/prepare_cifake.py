from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random

cifake_root = Path('data/cifake')
real_dir = cifake_root / 'REAL'
fake_dir = cifake_root / 'FAKE'

if not real_dir.exists() or not fake_dir.exists():
    print("Download CIFAKE dataset first:")
    print("kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake/")
    exit(1)

real_images = list(real_dir.glob('*.png'))[:3000]
fake_images = list(fake_dir.glob('*.png'))[:3000]

print(f"Found {len(real_images)} real, {len(fake_images)} fake")

real_train, real_val = train_test_split(real_images, test_size=0.2, random_state=42)
fake_train, fake_val = train_test_split(fake_images, test_size=0.2, random_state=42)

train_data = [(str(p), 0) for p in real_train] + [(str(p), 1) for p in fake_train]
val_data = [(str(p), 0) for p in real_val] + [(str(p), 1) for p in fake_val]

random.shuffle(train_data)
random.shuffle(val_data)

import pickle
with open('data/cifake_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('data/cifake_val.pkl', 'wb') as f:
    pickle.dump(val_data, f)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")
print("Saved to data/cifake_train.pkl and data/cifake_val.pkl")
