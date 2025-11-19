import gradio as gr
import yaml
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import Evaluator
from src.feature_extractor import MultiModalFeatureExtractor

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

evaluator = Evaluator(config, 'models/best_model.pt', device='cpu')
feature_extractor = MultiModalFeatureExtractor(image_size=224, device='cpu')

def get_interpretation(value, metric_type):
    interpretations = {
        'freq_ratio': {
            'high': (100, "🔴 Very High - Strong indicator of AI over-smoothing"),
            'medium': (50, "🟡 Moderate - Some smoothing detected"),
            'low': (0, "🟢 Normal - Natural frequency distribution")
        },
        'edge_density': {
            'high': (50, "🟢 High - Natural sharp transitions"),
            'medium': (20, "🟡 Moderate - Some edge preservation"),
            'low': (0, "🔴 Low - Artificial smoothing detected")
        },
        'smoothness_ratio': {
            'high': (0.8, "🔴 High - Unnatural uniformity (AI-like)"),
            'medium': (0.5, "🟡 Moderate - Some uniformity detected"),
            'low': (0, "🟢 Low - Natural variation")
        },
        'texture_entropy': {
            'high': (6, "🟢 High - Rich natural texture"),
            'medium': (4, "🟡 Moderate - Some texture detail"),
            'low': (0, "🔴 Low - Flat/synthetic texture")
        }
    }
    
    thresholds = interpretations[metric_type]
    if value >= thresholds['high'][0]:
        return thresholds['high'][1]
    elif value >= thresholds['medium'][0]:
        return thresholds['medium'][1]
    else:
        return thresholds['low'][1]

def generate_summary(label, confidence, features):
    if label == 1:
        certainty = "highly likely" if confidence > 0.6 else "likely" if confidence > 0.4 else "possibly"
        summary = f"🤖 **This image is {certainty} AI-generated.** "
        
        indicators = []
        if features['frequency'][2] > 100:
            indicators.append("excessive smoothing")
        if features['texture'][1] < 30:
            indicators.append("reduced edge detail")
        if features['semantic'][2] > 0.8:
            indicators.append("unnatural uniformity")
        
        if indicators:
            summary += f"Key indicators: {', '.join(indicators)}."
    else:
        certainty = "highly likely" if confidence < 0.4 else "likely" if confidence < 0.6 else "possibly"
        summary = f"📸 **This image is {certainty} a real photograph.** "
        
        indicators = []
        if features['frequency'][2] < 50:
            indicators.append("natural frequency distribution")
        if features['texture'][1] > 40:
            indicators.append("sharp edge detail")
        if features['semantic'][2] < 0.5:
            indicators.append("natural variation")
        
        if indicators:
            summary += f"Key indicators: {', '.join(indicators)}."
    
    return summary

def analyze_image(image):
    if image is None:
        return "No image provided", ""
    
    temp_path = "temp_upload.jpg"
    Image.fromarray(image).save(temp_path)
    
    label, confidence = evaluator.predict(temp_path)
    features = feature_extractor.extract_all_features(temp_path)
    
    result_emoji = "🤖" if label == 1 else "📸"
    result_text = "AI-Generated (Slop)" if label == 1 else "Real Image"
    confidence_pct = confidence * 100
    
    summary = generate_summary(label, confidence, features)
    
    freq_ratio_interp = get_interpretation(features['frequency'][2], 'freq_ratio')
    edge_density_interp = get_interpretation(features['texture'][1], 'edge_density')
    smoothness_interp = get_interpretation(features['semantic'][2], 'smoothness_ratio')
    texture_entropy_interp = get_interpretation(features['texture'][3], 'texture_entropy')
    
    analysis = f"""
## {result_emoji} Prediction: {result_text}
**Confidence Score:** {confidence_pct:.1f}%

---

{summary}

---

### 📊 Detailed Feature Analysis

#### 🌊 Frequency Domain Analysis
Analyzes the distribution of frequencies in the image. AI-generated images often have unnatural frequency patterns due to over-smoothing.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Low Frequency Energy | {features['frequency'][0]:.2f} | Total energy in smooth/gradual changes |
| High Frequency Energy | {features['frequency'][1]:.2f} | Total energy in sharp/detailed areas |
| **Frequency Ratio** | **{features['frequency'][2]:.2f}** | {freq_ratio_interp} |
| Frequency Entropy | {features['frequency'][3]:.2f} | Complexity of frequency distribution |

> **Higher ratio = More AI-like** (AI tends to over-smooth, reducing high frequencies)

---

#### 🎨 Texture Analysis
Examines edge sharpness and texture patterns. Real photos have richer, more chaotic texture.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Contrast | {features['texture'][0]:.2f} | Overall contrast intensity |
| **Edge Density** | **{features['texture'][1]:.2f}** | {edge_density_interp} |
| Gradient Std | {features['texture'][2]:.2f} | Variation in gradients |
| **Texture Entropy** | **{features['texture'][3]:.2f}** | {texture_entropy_interp} |

> **Lower edge density + entropy = More AI-like** (AI smooths out natural texture complexity)

---

#### 🔍 Semantic Artifact Detection
Analyzes spatial consistency and block-level smoothness patterns.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Block Smoothness Mean | {features['semantic'][0]:.2f} | Average patch variation |
| Block Smoothness Std | {features['semantic'][1]:.2f} | Consistency across patches |
| **Smoothness Ratio** | **{features['semantic'][2]:.2f}** | {smoothness_interp} |

> **Higher ratio = More AI-like** (Unnatural uniformity across image regions)

---

### 🧠 Model Decision Process
The RL agent learned to weight these features dynamically:
- Frequency features: 24.5%
- Texture features: 20.9%
- Perceptual features: 26.9%
- Semantic features: 27.8%
"""
    
    json_output = {
        "prediction": {
            "label": "ai_generated" if label == 1 else "real",
            "confidence": round(float(confidence), 3),
            "timestamp": datetime.now().isoformat()
        },
        "features": {
            "frequency": {
                "low_freq_energy": round(float(features['frequency'][0]), 2),
                "high_freq_energy": round(float(features['frequency'][1]), 2),
                "freq_ratio": round(float(features['frequency'][2]), 2),
                "freq_entropy": round(float(features['frequency'][3]), 2)
            },
            "texture": {
                "contrast": round(float(features['texture'][0]), 2),
                "edge_density": round(float(features['texture'][1]), 2),
                "gradient_std": round(float(features['texture'][2]), 2),
                "texture_entropy": round(float(features['texture'][3]), 2)
            },
            "semantic": {
                "block_smoothness_mean": round(float(features['semantic'][0]), 2),
                "block_smoothness_std": round(float(features['semantic'][1]), 2),
                "smoothness_ratio": round(float(features['semantic'][2]), 2)
            }
        },
        "indicators": {
            "frequency_ratio_high": bool(features['frequency'][2] > 100),
            "edge_density_low": bool(features['texture'][1] < 30),
            "smoothness_ratio_high": bool(features['semantic'][2] > 0.8)
        }
    }
    
    json_str = json.dumps(json_output, indent=2)
    
    return analysis, json_str

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(label="Upload Image for Analysis"),
    outputs=[
        gr.Markdown(label="Analysis Report"),
        gr.Code(label="JSON Export", language="json")
    ],
    title="🔬 AI Slop Detector",
    description="""
    ### Advanced AI-Generated Image Detection System
    
    This tool uses **Reinforcement Learning** and **Multi-Modal Feature Analysis** to detect AI-generated images.
    
    **How it works:**
    1. Extracts 75 features across frequency, texture, perceptual, and semantic domains
    2. RL agent adaptively weights features based on learned patterns
    3. Achieves **89% accuracy** (91% on real images, 87% on AI-generated)
    
    **Upload any image to get:**
    - Binary classification (Real vs AI-generated)
    - Confidence score
    - Detailed feature breakdown with interpretations
    - JSON export for API integration
    """,
    examples=[
        ["data/cache/base_0.jpg"],
        ["data/cache/slop_0.jpg"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
