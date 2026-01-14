"""
Project templates for different AI project types.
"""

from pathlib import Path
from typing import Dict
from yctl.utils import write_file


class ProjectTemplate:
    """Base class for project templates."""
    
    def __init__(self, project_name: str, project_path: Path):
        self.project_name = project_name
        self.project_path = project_path
    
    def create_structure(self) -> None:
        """Create the project directory structure."""
        raise NotImplementedError
    
    def get_requirements(self) -> str:
        """Get the requirements.txt content."""
        raise NotImplementedError
    
    def get_readme(self) -> str:
        """Get the README.md content."""
        raise NotImplementedError


class NLPTemplate(ProjectTemplate):
    """Template for NLP projects."""
    
    def create_structure(self) -> None:
        dirs = [
            "data/raw",
            "data/processed",
            "notebooks",
            "src/models",
            "src/preprocessing",
            "src/utils",
            "tests",
            "configs",
            "outputs/models",
            "outputs/logs",
        ]
        for dir_path in dirs:
            (self.project_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        for init_path in ["src", "src/models", "src/preprocessing", "src/utils", "tests"]:
            write_file(self.project_path / init_path / "__init__.py", "")
        
        # Create main training script
        train_script = '''"""
Main training script for NLP model.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train NLP model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Training with data from: {args.data}")
    # Add your training logic here


if __name__ == "__main__":
    main()
'''
        write_file(self.project_path / "src" / "train.py", train_script)
        
        # Create config file
        config = '''# Model Configuration
model:
  name: "transformer"
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  dropout: 0.1

# Training Configuration
training:
  batch_size: 32
  epochs: 10
  learning_rate: 1e-4
  warmup_steps: 1000
  
# Data Configuration
data:
  max_length: 512
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
'''
        write_file(self.project_path / "configs" / "config.yaml", config)
    
    def get_requirements(self) -> str:
        return """# Core
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0

# NLP Libraries
spacy>=3.6.0
nltk>=3.8.0
sentencepiece>=0.1.99

# Utilities
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0

# Experiment Tracking
wandb>=0.15.0
tensorboard>=2.13.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
"""
    
    def get_readme(self) -> str:
        return f"""# {self.project_name}

NLP project created with yctl.

## Project Structure

```
{self.project_name}/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Preprocessed data
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── models/          # Model architectures
│   ├── preprocessing/   # Data preprocessing
│   ├── utils/           # Utility functions
│   └── train.py         # Main training script
├── tests/               # Unit tests
├── configs/             # Configuration files
└── outputs/
    ├── models/          # Saved models
    └── logs/            # Training logs
```

## Setup

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Train a model:
```bash
python src/train.py --data data/processed/train.csv --epochs 10
```

## Best Practices

- Keep raw data immutable in `data/raw/`
- Version your datasets and models
- Use configuration files for hyperparameters
- Log experiments with wandb or tensorboard
- Write tests for critical components
- Use pre-commit hooks for code quality

## Common NLP Tasks

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Extract entities from text
- **Question Answering**: Build QA systems
- **Text Generation**: GPT-style models
- **Machine Translation**: Seq2seq models
- **Summarization**: Abstractive/extractive summarization

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [spaCy Documentation](https://spacy.io/usage)
- [Papers with Code - NLP](https://paperswithcode.com/area/natural-language-processing)
"""


class CVTemplate(ProjectTemplate):
    """Template for Computer Vision projects."""
    
    def create_structure(self) -> None:
        dirs = [
            "data/raw/train",
            "data/raw/val",
            "data/raw/test",
            "data/processed",
            "notebooks",
            "src/models",
            "src/data",
            "src/augmentation",
            "src/utils",
            "tests",
            "configs",
            "outputs/models",
            "outputs/logs",
            "outputs/visualizations",
        ]
        for dir_path in dirs:
            (self.project_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        for init_path in ["src", "src/models", "src/data", "src/augmentation", "src/utils", "tests"]:
            write_file(self.project_path / init_path / "__init__.py", "")
        
        # Create main training script
        train_script = '''"""
Main training script for CV model.
"""

import argparse
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train CV model")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training with data from: {args.data}")
    # Add your training logic here


if __name__ == "__main__":
    main()
'''
        write_file(self.project_path / "src" / "train.py", train_script)
        
        # Create config file
        config = '''# Model Configuration
model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 10
  dropout: 0.5

# Training Configuration
training:
  batch_size: 64
  epochs: 50
  learning_rate: 1e-3
  optimizer: "adam"
  scheduler: "cosine"
  
# Data Configuration
data:
  image_size: 224
  num_workers: 4
  augmentation: true
  
# Augmentation
augmentation:
  random_flip: true
  random_rotation: 15
  color_jitter: true
  normalize: true
'''
        write_file(self.project_path / "configs" / "config.yaml", config)
    
    def get_requirements(self) -> str:
        return """# Core
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# CV Libraries
opencv-python>=4.8.0
Pillow>=10.0.0
albumentations>=1.3.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment Tracking
wandb>=0.15.0
tensorboard>=2.13.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
"""
    
    def get_readme(self) -> str:
        return f"""# {self.project_name}

Computer Vision project created with yctl.

## Project Structure

```
{self.project_name}/
├── data/
│   ├── raw/
│   │   ├── train/       # Training images
│   │   ├── val/         # Validation images
│   │   └── test/        # Test images
│   └── processed/       # Preprocessed data
├── notebooks/           # Jupyter notebooks
├── src/
│   ├── models/         # Model architectures
│   ├── data/           # Dataset classes
│   ├── augmentation/   # Data augmentation
│   ├── utils/          # Utility functions
│   └── train.py        # Main training script
├── tests/              # Unit tests
├── configs/            # Configuration files
└── outputs/
    ├── models/         # Saved models
    ├── logs/           # Training logs
    └── visualizations/ # Plots and visualizations
```

## Setup

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python src/train.py --data data/raw --epochs 50 --batch-size 64
```

## Best Practices

- Organize images in class-based folders
- Use data augmentation to prevent overfitting
- Monitor training with tensorboard
- Save checkpoints regularly
- Use mixed precision training for speed
- Validate on a separate dataset
- Use pre-trained models when possible

## Common CV Tasks

- **Image Classification**: ResNet, EfficientNet, Vision Transformers
- **Object Detection**: YOLO, Faster R-CNN, DETR
- **Semantic Segmentation**: U-Net, DeepLab, Mask R-CNN
- **Instance Segmentation**: Mask R-CNN, YOLACT
- **Image Generation**: GANs, Diffusion Models
- **Face Recognition**: ArcFace, FaceNet

## Resources

- [PyTorch Vision Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [Papers with Code - CV](https://paperswithcode.com/area/computer-vision)
"""


class MLTemplate(ProjectTemplate):
    """Template for general ML projects."""
    
    def create_structure(self) -> None:
        dirs = [
            "data/raw",
            "data/processed",
            "notebooks",
            "src/models",
            "src/features",
            "src/utils",
            "tests",
            "configs",
            "outputs/models",
            "outputs/logs",
            "outputs/predictions",
        ]
        for dir_path in dirs:
            (self.project_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        for init_path in ["src", "src/models", "src/features", "src/utils", "tests"]:
            write_file(self.project_path / init_path / "__init__.py", "")
        
        # Create main training script
        train_script = '''"""
Main training script for ML model.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded data: {df.shape}")
    
    # Split features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    # Add your training logic here


if __name__ == "__main__":
    main()
'''
        write_file(self.project_path / "src" / "train.py", train_script)
        
        # Create config file
        config = '''# Model Configuration
model:
  type: "gradient_boosting"  # random_forest, xgboost, lightgbm, etc.
  params:
    n_estimators: 100
    max_depth: 10
    learning_rate: 0.1
    
# Feature Engineering
features:
  scaling: "standard"  # standard, minmax, robust
  encoding: "onehot"   # onehot, label, target
  handle_missing: "median"  # mean, median, drop
  
# Cross-Validation
cv:
  n_splits: 5
  shuffle: true
  random_state: 42
'''
        write_file(self.project_path / "configs" / "config.yaml", config)
    
    def get_requirements(self) -> str:
        return """# Core ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# Feature Engineering
feature-engine>=1.6.0
category-encoders>=2.6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Model Interpretation
shap>=0.42.0
lime>=0.2.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0

# Experiment Tracking
mlflow>=2.5.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
"""
    
    def get_readme(self) -> str:
        return f"""# {self.project_name}

Machine Learning project created with yctl.

## Project Structure

```
{self.project_name}/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed features
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── models/          # Model implementations
│   ├── features/        # Feature engineering
│   ├── utils/           # Utility functions
│   └── train.py         # Main training script
├── tests/               # Unit tests
├── configs/             # Configuration files
└── outputs/
    ├── models/          # Saved models
    ├── logs/            # Training logs
    └── predictions/     # Model predictions
```

## Setup

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python src/train.py --data data/processed/train.csv --target target_column
```

## Best Practices

- Start with exploratory data analysis (EDA)
- Handle missing values appropriately
- Scale/normalize features
- Use cross-validation for model selection
- Track experiments with MLflow
- Interpret models with SHAP/LIME
- Version your datasets and models
- Document feature engineering steps

## ML Workflow

1. **Data Collection**: Gather and store raw data
2. **EDA**: Understand data distribution and relationships
3. **Feature Engineering**: Create meaningful features
4. **Model Selection**: Try multiple algorithms
5. **Hyperparameter Tuning**: Optimize model parameters
6. **Evaluation**: Assess model performance
7. **Deployment**: Serve model predictions

## Common ML Tasks

- **Classification**: Logistic Regression, Random Forest, XGBoost
- **Regression**: Linear Regression, Ridge, Lasso, Gradient Boosting
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest, One-Class SVM

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
"""


class ResearchTemplate(ProjectTemplate):
    """Template for research projects."""
    
    def create_structure(self) -> None:
        dirs = [
            "data/raw",
            "data/processed",
            "experiments",
            "notebooks",
            "src/models",
            "src/utils",
            "tests",
            "papers",
            "results/figures",
            "results/tables",
            "results/checkpoints",
        ]
        for dir_path in dirs:
            (self.project_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        for init_path in ["src", "src/models", "src/utils", "tests"]:
            write_file(self.project_path / init_path / "__init__.py", "")
        
        # Create experiment script
        experiment_script = '''"""
Experiment runner for research.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{args.name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        "name": args.name,
        "timestamp": timestamp,
        "seed": args.seed,
    }
    
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Running experiment: {args.name}")
    print(f"Results will be saved to: {exp_dir}")
    
    # Add your experiment logic here


if __name__ == "__main__":
    main()
'''
        write_file(self.project_path / "src" / "experiment.py", experiment_script)
        
        # Create experiment template
        exp_template = '''# Experiment: [Name]

**Date**: [YYYY-MM-DD]
**Researcher**: [Your Name]

## Hypothesis

[State your hypothesis here]

## Methodology

### Data
- Dataset: [Name and source]
- Size: [Number of samples]
- Features: [Description]

### Model
- Architecture: [Model details]
- Hyperparameters: [List key parameters]

### Training
- Optimizer: [e.g., Adam]
- Learning rate: [value]
- Batch size: [value]
- Epochs: [value]

## Results

### Quantitative

| Metric | Value |
|--------|-------|
| Accuracy | X.XX% |
| Precision | X.XX% |
| Recall | X.XX% |
| F1-Score | X.XX% |

### Qualitative

[Observations and insights]

## Analysis

[Detailed analysis of results]

## Conclusions

[What did you learn?]

## Next Steps

- [ ] Action item 1
- [ ] Action item 2
- [ ] Action item 3

## References

1. [Paper 1]
2. [Paper 2]
'''
        write_file(self.project_path / "experiments" / "template.md", exp_template)
    
    def get_requirements(self) -> str:
        return """# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Experiment Tracking
wandb>=0.15.0
tensorboard>=2.13.0
mlflow>=2.5.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
jupyter>=1.0.0
ipykernel>=6.25.0

# Paper Writing
jupytext>=1.15.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
"""
    
    def get_readme(self) -> str:
        return f"""# {self.project_name}

Research project created with yctl.

## Project Structure

```
{self.project_name}/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed data
├── experiments/          # Experiment logs and configs
├── notebooks/            # Research notebooks
├── src/
│   ├── models/          # Model implementations
│   ├── utils/           # Utility functions
│   └── experiment.py    # Experiment runner
├── tests/               # Unit tests
├── papers/              # Related papers and notes
└── results/
    ├── figures/         # Plots and visualizations
    ├── tables/          # Result tables
    └── checkpoints/     # Model checkpoints
```

## Setup

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
python src/experiment.py --name exp_001 --config configs/baseline.yaml
```

## Research Workflow

1. **Literature Review**: Read and summarize related papers
2. **Hypothesis Formation**: Define clear research questions
3. **Experimental Design**: Plan experiments systematically
4. **Implementation**: Code models and experiments
5. **Execution**: Run experiments with proper logging
6. **Analysis**: Analyze results rigorously
7. **Documentation**: Document findings clearly
8. **Iteration**: Refine based on results

## Best Practices

- **Reproducibility**: Set random seeds, version everything
- **Documentation**: Keep detailed experiment logs
- **Version Control**: Commit code and configs regularly
- **Ablation Studies**: Test one change at a time
- **Statistical Significance**: Run multiple seeds
- **Visualization**: Plot learning curves and results
- **Code Quality**: Write clean, tested code
- **Paper Reading**: Stay updated with latest research

## Experiment Tracking

Use the experiment template in `experiments/template.md` for each experiment.

Track experiments with:
- Weights & Biases (wandb)
- TensorBoard
- MLflow

## Writing Papers

1. Keep a research journal
2. Document experiments immediately
3. Create figures as you go
4. Write incrementally
5. Get feedback early

## Resources

- [Papers with Code](https://paperswithcode.com/)
- [arXiv](https://arxiv.org/)
- [Google Scholar](https://scholar.google.com/)
- [Connected Papers](https://www.connectedpapers.com/)
- [Semantic Scholar](https://www.semanticscholar.org/)
"""


# Template registry
TEMPLATES: Dict[str, type[ProjectTemplate]] = {
    "nlp": NLPTemplate,
    "cv": CVTemplate,
    "ml": MLTemplate,
    "research": ResearchTemplate,
}


def get_template(project_type: str, project_name: str, project_path: Path) -> ProjectTemplate:
    """Get a project template instance."""
    template_class = TEMPLATES.get(project_type)
    if not template_class:
        raise ValueError(f"Unknown project type: {project_type}")
    return template_class(project_name, project_path)
