import argparse
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    evaluator = Evaluator(config, 'models/best_model.pt', device=args.device)
    label, confidence = evaluator.predict(args.image)
    
    result = "AI-generated (slop)" if label == 1 else "Real"
    print(f"{result} (confidence: {confidence:.3f})")

if __name__ == '__main__':
    main()
