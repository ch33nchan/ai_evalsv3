# RL-Based Multi-Modal AI Slop Detection

## Abstract
Novel approach using Deep Q-Learning for adaptive feature weighting in AI-generated image detection. Achieves 90% accuracy through dynamic fusion of frequency, texture, perceptual, and semantic features.

## 1. Introduction
- Problem: Static classifiers fail to adapt to evolving AI generation techniques
- Solution: RL agent learns optimal feature importance dynamically
- Contribution: First RL-based approach for slop detection

## 2. Method
### 2.1 Feature Extraction
- Frequency domain (DCT analysis)
- Texture features (edge density, gradients)
- Perceptual embeddings (ResNet50)
- Semantic artifacts (block smoothness)

### 2.2 RL Framework
- State: 75-dim multi-modal feature vector
- Action: 10 discrete actions mapping to binary classification
- Reward: Correctness + feature quality + confidence penalty
- Algorithm: DQN with target network

## 3. Experiments
- Dataset: 600 synthetic images (base + corrupted)
- Training: 100 episodes, 500 samples
- Results: 90% validation accuracy
- Feature importance: Balanced across modalities (20-28% each)

## 4. Results
- Real image detection: 92%
- AI slop detection: 88%
- Outperforms static weighting baseline

## 5. Future Work
- Test on real generative models (DALL-E, Midjourney, Stable Diffusion)
- Transfer learning to new AI architectures
- Online adaptation to emerging slop patterns
