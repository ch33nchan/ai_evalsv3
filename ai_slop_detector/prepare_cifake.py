from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import pickle

cifake_root = Path('data/cifake')
train_real = cifake_root / 'train' / 'REAL'
train_fake = cifake_root / 'train' / 'FAKE'
test_real = cifake_root / 'test' / 'REAL'
test_fake = cifake_root / 'test' / 'FAKE'

real_images = []
fake_images = []

for ext in ['*.png', '*.jpg', '*.jpeg']:
    if train_real.exists():
        real_images.extend(list(train_real.glob(ext)))
    if test_real.exists():
        real_images.extend(list(test_real.glob(ext)))
    if train_fake.exists():
        fake_images.extend(list(train_fake.glob(ext)))
    if test_fake.exists():
        fake_images.extend(list(test_fake.glob(ext)))

print(f"Found {len(real_images)} real, {len(fake_images)} fake")

if len(real_images) == 0 or len(fake_images) == 0:
    print("No images found. Check directories:")
    print(f"  {train_real}: exists={train_real.exists()}")
    print(f"  {train_fake}: exists={train_fake.exists()}")
    exit(1)

real_images = real_images[:3000]
fake_images = fake_images[:3000]

real_train, real_val = train_test_split(real_images, test_size=0.2, random_state=42)
fake_train, fake_val = train_test_split(fake_images, test_size=0.2, random_state=42)

train_data = [(str(p), 0) for p in real_train] + [(str(p), 1) for p in fake_train]
val_data = [(str(p), 0) for p in real_val] + [(str(p), 1) for p in fake_val]

random.shuffle(train_data)
random.shuffle(val_data)

with open('data/cifake_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('data/cifake_val.pkl', 'wb') as f:
    pickle.dump(val_data, f)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")
