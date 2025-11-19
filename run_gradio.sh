#!/bin/bash

cd ai_slop_detector
source venv/bin/activate

pip3 install gradio==4.8.0

python3 app.py
