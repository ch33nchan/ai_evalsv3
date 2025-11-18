import argparse
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import SlopDatasetGenerator
from src.train import Trainer
from src.evaluate import Evaluator

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'predict'])
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default='models/best_model.pt')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    Path('models').mkdir(exist_ok=True)
    
    if args.mode == 'train':
        print("Generating dataset...")
        dataset_gen = SlopDatasetGenerator(cache_dir=config['data']['cache_dir'])
        train_data, val_data = dataset_gen.create_dataset(
            train_samples=config['data']['train_samples'],
            val_samples=config['data']['val_samples']
        )
        
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        print("Training model...")
        trainer = Trainer(config, device=args.device)
        history = trainer.train(train_data, val_data, config['training']['num_episodes'])
        
        print(f"Training complete. Best val accuracy: {max(history['val_acc']):.3f}")
        
    elif args.mode == 'eval':
        print("Loading dataset for evaluation...")
        dataset_gen = SlopDatasetGenerator(cache_dir=config['data']['cache_dir'])
        _, val_data = dataset_gen.create_dataset(
            train_samples=0,
            val_samples=config['data']['val_samples']
        )
        
        evaluator = Evaluator(config, args.model, device=args.device)
        results = evaluator.evaluate_dataset(val_data)
        
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Correct: {results['correct']}/{results['total']}")
        
    elif args.mode == 'predict':
        if args.image is None:
            print("Error: --image required for predict mode")
            return
        
        evaluator = Evaluator(config, args.model, device=args.device)
        label, confidence = evaluator.predict(args.image)
        
        label_name = "AI-generated (slop)" if label == 1 else "Real"
        print(f"Prediction: {label_name}, Confidence: {confidence:.3f}")

if __name__ == '__main__':
    main()
