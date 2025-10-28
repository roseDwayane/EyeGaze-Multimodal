
# Dataprepare - I/O
Eye-Gaze Data Preparation — I/O Module

Overview
This module is part of my thesis project on eye-tracking data generation and classification using vision-language models
At this stage, the focus is on the Data Preparation I/O pipeline, which handles:
- Loading JSON annotation files
- Reading paired images from root directories
- Applying basic image enhancement
- Concatenating two images vertically
- Exporting a CSV manifest
- Saving preview images for inspection


# Dataprepare - Fusion
Eye-Gaze Data Preparation — Fusion Module 

Overview
The Fusion module handles: 
- Concatenating two eye-tracking visualization images vertically 
- Building sample datasets from JSON metadata with category balancing 
- Validating image file paths 
- Preparing data for vision-language model input 


# Result - Similarity
Overview
The Similarity module handles: 
- Calculating semantic similarity between reference and generated descriptions - Using TF-IDF vectorization and cosine similarity metrics 
- Processing multiple description styles and experimental conditions 
- Building comprehensive comparison results with metadata 
- Computing statistical summaries (mean, max, min, distribution) 
- Exporting timestamped similarity analysis reports 


# Experiment - Main
Overview
The main.py handles: 
- Training GIT (Generative Image-to-text Transformer) model on eye-tracking visualizations 
- Processing multiple experimental conditions across 6 ROOT directories 
- Fine-tuning with 3 description styles (formal, lively, mixed) 
- Generating natural language descriptions with multiple strategies 
- Evaluating generation quality and diversity 
- Exporting results for similarity analysis --- 

