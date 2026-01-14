# Quick Start Guide

## Installation

### 1. Install yctl

```bash
# Navigate to the yctl-cli directory
cd /home/mango/Coding/yctl-cli

# Install in development mode (recommended for development)
pip install -e .

# OR install normally
pip install .
```

### 2. Verify Installation

```bash
yctl --version
yctl --help
```

## First Steps

### Check Your System

```bash
yctl doctor
```

This will verify:
- Python version
- pip and venv
- GPU and CUDA
- Development tools

### Create Your First Project

```bash
# NLP Project
yctl init nlp my-first-nlp-project

# Navigate to the project
cd my-first-nlp-project

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Inspect a Dataset

```bash
# Create a sample CSV file
cat > sample.csv << EOF
name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Engineering
Charlie,35,70000,Sales
David,28,55000,Marketing
Eve,32,65000,Engineering
EOF

# Inspect it
yctl inspect sample.csv
```

### Analyze an AI Idea

```bash
yctl think "build a chatbot for customer support"
```

## Example Workflows

### Workflow 1: NLP Sentiment Analysis

```bash
# 1. Analyze the idea
yctl think "sentiment analysis for product reviews"

# 2. Create the project
yctl init nlp product-sentiment

# 3. Setup
cd product-sentiment
source venv/bin/activate
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Inspect your data
yctl inspect ../data/reviews.csv

# 6. Start coding!
# Edit src/train.py and start training
```

### Workflow 2: Computer Vision

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
# Place images in data/raw/train/class_name/
# data/raw/train/healthy/
# data/raw/train/diseased/

# 5. Start training
python src/train.py --data data/raw --epochs 50
```

### Workflow 3: Machine Learning

```bash
# 1. Create the project
yctl init ml house-price-prediction

# 2. Setup
cd house-price-prediction
source venv/bin/activate
pip install -r requirements.txt

# 3. Inspect your dataset
yctl inspect data/raw/housing.csv

# 4. Start with EDA in notebooks
jupyter notebook notebooks/

# 5. Train your model
python src/train.py --data data/processed/train.csv --target price
```

## Tips and Tricks

### Use Virtual Environments

Always activate your virtual environment:
```bash
source venv/bin/activate
```

### Track Experiments

Use Weights & Biases:
```bash
pip install wandb
wandb login
```

### GPU Training

Check GPU availability:
```bash
yctl doctor
```

Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Code Quality

Format your code:
```bash
pip install black
black src/
```

Run linting:
```bash
pip install flake8
flake8 src/
```

### Testing

Write tests:
```bash
pip install pytest
pytest tests/
```

## Common Issues

### Issue: "yctl: command not found"

**Solution:**
```bash
# Make sure you installed yctl
pip install -e .

# Or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: GPU not detected

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot
```

### Issue: CUDA not available in PyTorch

**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

1. Read the full README.md
2. Explore the examples/ directory
3. Check out the project templates
4. Start building your AI projects!

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai](https://www.fast.ai/)
- [Papers with Code](https://paperswithcode.com/)
- [Kaggle Learn](https://www.kaggle.com/learn)

---

**Happy coding! 🚀**
