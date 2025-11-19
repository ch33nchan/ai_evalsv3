#!/bin/bash

cd ai_slop_detector
source venv/bin/activate

echo "=== Model Evaluation ==="
python3 main.py --mode eval --device cuda --model models/best_model.pt

echo ""
echo "=== Testing on Sample Images ==="

# Test on a real image
REAL_IMG=$(ls data/cache/real_*.jpg 2>/dev/null | head -1)
if [ -n "$REAL_IMG" ]; then
    echo "Testing real image: $REAL_IMG"
    python3 main.py --mode predict --device cuda --image "$REAL_IMG"
fi

# Test on a synthetic image
SYNTH_IMG=$(ls data/cache/synthetic_*.jpg 2>/dev/null | head -1)
if [ -n "$SYNTH_IMG" ]; then
    echo "Testing synthetic image: $SYNTH_IMG"
    python3 main.py --mode predict --device cuda --image "$SYNTH_IMG"
fi

deactivate
