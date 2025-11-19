import gradio as gr
import yaml
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import Evaluator
from src.feature_extractor import MultiModalFeatureExtractor

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

evaluator = Evaluator(config, 'models/best_model.pt', device='cpu')
feature_extractor = MultiModalFeatureExtractor(image_size=224, device='cpu')

def analyze_image(image):
    if image is None:
        return "No image provided", None
    
    temp_path = "temp_upload.jpg"
    Image.fromarray(image).save(temp_path)
    
    label, confidence = evaluator.predict(temp_path)
    
    features = feature_extractor.extract_all_features(temp_path)
    
    result = "🤖 AI-Generated (Slop)" if label == 1 else "📸 Real Image"
    confidence_pct = confidence * 100
    
    feature_analysis = f"""
### Feature Analysis

**Frequency Domain:**
- Low freq energy: {features['frequency'][0]:.2f}
- High freq energy: {features['frequency'][1]:.2f}
- Freq ratio: {features['frequency'][2]:.2f}
- Freq entropy: {features['frequency'][3]:.2f}

**Texture:**
- Contrast: {features['texture'][0]:.2f}
- Edge density: {features['texture'][1]:.2f}
- Gradient std: {features['texture'][2]:.2f}
- Texture entropy: {features['texture'][3]:.2f}

**Semantic Artifacts:**
- Block smoothness mean: {features['semantic'][0]:.2f}
- Block smoothness std: {features['semantic'][1]:.2f}
- Smoothness ratio: {features['semantic'][2]:.2f}
"""
    
    return f"# {result}\n**Confidence:** {confidence_pct:.1f}%", feature_analysis

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(label="Upload Image"),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Markdown(label="Technical Analysis")
    ],
    title="AI Slop Detector",
    description="""
    Detects AI-generated images using multi-modal feature analysis and reinforcement learning.
    
    **Features:**
    - Frequency domain analysis (DCT artifacts)
    - Texture analysis (edges, gradients)
    - Perceptual embeddings (ResNet50)
    - Semantic artifact detection
    
    **Accuracy:** 89% on validation set (91% real, 87% AI-generated)
    """,
    examples=[
        ["data/cache/base_0.jpg"],
        ["data/cache/slop_0.jpg"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
