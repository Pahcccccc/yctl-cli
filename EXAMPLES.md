# yctl Examples

This document provides comprehensive examples of using yctl for various AI/ML workflows.

## Table of Contents

1. [Basic Commands](#basic-commands)
2. [NLP Projects](#nlp-projects)
3. [Computer Vision Projects](#computer-vision-projects)
4. [Machine Learning Projects](#machine-learning-projects)
5. [Research Projects](#research-projects)
6. [Dataset Inspection](#dataset-inspection)
7. [System Diagnostics](#system-diagnostics)
8. [AI Idea Analysis](#ai-idea-analysis)

---

## Basic Commands

### Check Version and Help

```bash
# Show version
yctl --version

# Show general help
yctl --help

# Show help for specific command
yctl init --help
yctl inspect --help
yctl doctor --help
yctl think --help
```

### System Health Check

```bash
# Run comprehensive system diagnostics
yctl doctor
```

**Output includes:**
- Python version check
- pip and venv availability
- GPU detection
- CUDA status
- PyTorch CUDA support
- Common development tools

---

## NLP Projects

### Example 1: Sentiment Analysis

```bash
# 1. Analyze the idea
yctl think "sentiment analysis for product reviews"

# 2. Create the project
yctl init nlp product-sentiment

# 3. Navigate and setup
cd product-sentiment
source venv/bin/activate
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Inspect your dataset
yctl inspect data/raw/reviews.csv

# 6. Start training
python src/train.py --data data/processed/train.csv --epochs 10
```

### Example 2: Question Answering System

```bash
# Create project
yctl init nlp qa-system

cd qa-system
source venv/bin/activate
pip install -r requirements.txt

# Your QA system code here
```

### Example 3: Text Classification

```bash
# Analyze the idea first
yctl think "multi-class text classification for news articles"

# Create project
yctl init nlp news-classifier

cd news-classifier
source venv/bin/activate
pip install -r requirements.txt
```

---

## Computer Vision Projects

### Example 1: Image Classification

```bash
# 1. Analyze the idea
yctl think "image classification for plant diseases"

# 2. Create the project
yctl init cv plant-disease-classifier

# 3. Setup
cd plant-disease-classifier
source venv/bin/activate
pip install -r requirements.txt

# 4. Organize your images
# Place images in class-based folders:
# data/raw/train/healthy/
# data/raw/train/diseased/
# data/raw/val/healthy/
# data/raw/val/diseased/

# 5. Train the model
python src/train.py --data data/raw --epochs 50 --batch-size 64
```

### Example 2: Object Detection

```bash
# Create project
yctl init cv object-detector

cd object-detector
source venv/bin/activate
pip install -r requirements.txt

# Install additional dependencies for object detection
pip install ultralytics  # For YOLO
```

### Example 3: Image Segmentation

```bash
# Analyze the idea
yctl think "semantic segmentation for autonomous driving"

# Create project
yctl init cv road-segmentation

cd road-segmentation
source venv/bin/activate
pip install -r requirements.txt
```

---

## Machine Learning Projects

### Example 1: House Price Prediction

```bash
# 1. Create the project
yctl init ml house-price-prediction

# 2. Setup
cd house-price-prediction
source venv/bin/activate
pip install -r requirements.txt

# 3. Inspect your dataset
yctl inspect data/raw/housing.csv

# 4. Start with EDA
jupyter notebook notebooks/

# 5. Train your model
python src/train.py --data data/processed/train.csv --target price
```

### Example 2: Customer Churn Prediction

```bash
# Analyze the idea
yctl think "predict customer churn using machine learning"

# Create project
yctl init ml churn-prediction

cd churn-prediction
source venv/bin/activate
pip install -r requirements.txt

# Inspect the dataset
yctl inspect data/raw/customers.csv
```

### Example 3: Recommendation System

```bash
# Create project
yctl init ml movie-recommender

cd movie-recommender
source venv/bin/activate
pip install -r requirements.txt

# Install additional libraries
pip install surprise implicit
```

---

## Research Projects

### Example 1: Novel Architecture Research

```bash
# 1. Create research project
yctl init research transformer-variant

# 2. Setup
cd transformer-variant
source venv/bin/activate
pip install -r requirements.txt

# 3. Use the experiment template
cp experiments/template.md experiments/exp_001_baseline.md

# 4. Run experiment
python src/experiment.py --name baseline --config configs/baseline.yaml --seed 42

# 5. Track with wandb
wandb login
# Add wandb.init() to your experiment code
```

### Example 2: Ablation Study

```bash
# Create research project
yctl init research attention-mechanism

cd attention-mechanism
source venv/bin/activate
pip install -r requirements.txt

# Run multiple experiments
python src/experiment.py --name exp_001_full_model --config configs/full.yaml
python src/experiment.py --name exp_002_no_attention --config configs/no_attention.yaml
python src/experiment.py --name exp_003_simple_attention --config configs/simple.yaml
```

---

## Dataset Inspection

### Example 1: CSV File

```bash
# Create a sample CSV
cat > sample.csv << EOF
name,age,salary,department,years_experience
Alice,25,50000,Engineering,2
Bob,30,60000,Engineering,5
Charlie,35,70000,Sales,8
David,28,55000,Marketing,3
Eve,32,65000,Engineering,6
EOF

# Inspect it
yctl inspect sample.csv
```

**Output includes:**
- Dataset overview (rows, columns, memory)
- Column types
- Missing values analysis
- Numeric statistics
- Categorical statistics
- Data quality issues
- Preprocessing suggestions
- Model recommendations

### Example 2: Large Dataset

```bash
# Inspect a large Parquet file
yctl inspect data/large_dataset.parquet
```

### Example 3: Excel File

```bash
# Inspect Excel file
yctl inspect data/sales_data.xlsx
```

---

## System Diagnostics

### Example 1: Pre-project Setup Check

```bash
# Before starting a new project, check your system
yctl doctor

# If issues are found, follow the suggested fixes
# Example: Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Example 2: GPU Troubleshooting

```bash
# Check if GPU is detected
yctl doctor

# Look for:
# - GPU status
# - CUDA availability
# - PyTorch CUDA support

# If GPU not detected, check NVIDIA drivers
nvidia-smi

# Install drivers if needed
sudo ubuntu-drivers autoinstall
sudo reboot
```

---

## AI Idea Analysis

### Example 1: NLP Idea

```bash
yctl think "sentiment analysis for customer reviews"
```

**Provides:**
- Feasibility: HIGH
- Complexity: INTERMEDIATE
- Roadmap with 6 steps
- Datasets: IMDB, Twitter Sentiment, Amazon Reviews
- Models: BERT, RoBERTa, DistilBERT, LSTM
- Tools: transformers, torch, scikit-learn
- Challenges and resources

### Example 2: Computer Vision Idea

```bash
yctl think "object detection for autonomous driving"
```

**Provides:**
- Feasibility assessment
- Complexity rating
- Detailed roadmap
- Datasets: COCO, KITTI, BDD100K
- Models: YOLO, Faster R-CNN, DETR
- Tools and libraries
- Potential challenges
- Learning resources

### Example 3: ML Idea

```bash
yctl think "time series forecasting for stock prices"
```

### Example 4: General Idea

```bash
yctl think "build an AI system to detect fake news"
```

---

## Complete Workflows

### Workflow 1: End-to-End NLP Project

```bash
# Step 1: System check
yctl doctor

# Step 2: Analyze idea
yctl think "sentiment analysis for movie reviews"

# Step 3: Create project
yctl init nlp movie-sentiment

# Step 4: Setup environment
cd movie-sentiment
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Step 5: Get data and inspect
# Download IMDB dataset
yctl inspect data/raw/imdb_reviews.csv

# Step 6: Start development
# Edit src/train.py
# Add your model code

# Step 7: Train
python src/train.py --data data/processed/train.csv --epochs 10

# Step 8: Evaluate
# Add evaluation code
```

### Workflow 2: End-to-End CV Project

```bash
# Step 1: Analyze idea
yctl think "image classification for medical diagnosis"

# Step 2: Create project
yctl init cv medical-image-classifier

# Step 3: Setup
cd medical-image-classifier
source venv/bin/activate
pip install -r requirements.txt

# Step 4: Organize data
# data/raw/train/normal/
# data/raw/train/abnormal/

# Step 5: Train with transfer learning
python src/train.py --data data/raw --epochs 50 --img-size 224

# Step 6: Evaluate and visualize
# Check outputs/visualizations/
```

### Workflow 3: Kaggle Competition

```bash
# Step 1: Analyze the competition
yctl think "predict house prices using regression"

# Step 2: Create project
yctl init ml kaggle-house-prices

# Step 3: Setup
cd kaggle-house-prices
source venv/bin/activate
pip install -r requirements.txt

# Step 4: Download data from Kaggle
# kaggle competitions download -c house-prices-advanced-regression-techniques

# Step 5: Inspect data
yctl inspect data/raw/train.csv
yctl inspect data/raw/test.csv

# Step 6: EDA in notebooks
jupyter notebook notebooks/

# Step 7: Feature engineering and training
python src/train.py --data data/raw/train.csv --target SalePrice

# Step 8: Make predictions
# python src/predict.py --model outputs/models/best_model.pkl --data data/raw/test.csv
```

---

## Advanced Usage

### Custom Project Types

You can modify the templates in `yctl/core/templates.py` to add custom project types.

### Integration with Other Tools

```bash
# Use with Git
cd my-project
git init
git add .
git commit -m "Initial commit from yctl"

# Use with Docker
# Add Dockerfile to your project
docker build -t my-ai-project .

# Use with DVC for data versioning
dvc init
dvc add data/raw/dataset.csv
```

### Batch Processing

```bash
# Inspect multiple datasets
for file in data/*.csv; do
    echo "Inspecting $file"
    yctl inspect "$file"
done
```

---

## Tips and Best Practices

1. **Always run `yctl doctor` before starting a new project**
2. **Use `yctl think` to validate your ideas before implementation**
3. **Inspect datasets before training to understand data quality**
4. **Keep virtual environments isolated per project**
5. **Use configuration files for hyperparameters**
6. **Track experiments with wandb or tensorboard**
7. **Version your data and models**
8. **Write tests for critical components**
9. **Document your experiments**
10. **Use GPU when available for deep learning**

---

## Troubleshooting

### Command not found

```bash
# Ensure yctl is installed
pip install -e .

# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### GPU not detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Install if needed
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Virtual environment issues

```bash
# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**For more information, see the [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)**
