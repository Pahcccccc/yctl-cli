# yctl - Complete Implementation Guide

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Architecture](#architecture)
4. [Commands Reference](#commands-reference)
5. [File Structure](#file-structure)
6. [Testing](#testing)
7. [Usage Examples](#usage-examples)
8. [Extending yctl](#extending-yctl)

---

## 📦 Project Overview

**yctl** is a comprehensive Personal AI Engineer CLI Tool built with Python, designed to streamline AI/ML development workflows on Ubuntu.

### Key Statistics
- **Total Python Files**: 13
- **Total Lines of Code**: ~3,500+
- **Commands**: 4 (init, inspect, doctor, think)
- **Project Templates**: 4 (NLP, CV, ML, Research)
- **Dependencies**: 8 core packages

### Core Technologies
- **Typer**: CLI framework with Rich integration
- **Rich**: Beautiful terminal output
- **Pandas/NumPy**: Data analysis
- **Scikit-learn**: ML utilities
- **psutil/GPUtil**: System monitoring

---

## 🚀 Installation

### Method 1: Development Install (Recommended)

```bash
cd /home/mango/Coding/yctl-cli
pip install -e .
```

### Method 2: Regular Install

```bash
cd /home/mango/Coding/yctl-cli
pip install .
```

### Method 3: From PyPI (Future)

```bash
pip install yctl
```

### Verification

```bash
# Check installation
yctl --help

# Run verification script
./install_and_verify.sh
```

---

## 🏗️ Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    yctl CLI (Typer)                     │
├─────────────────────────────────────────────────────────┤
│  Commands Layer                                         │
│  ├── init.py      (Project Initialization)             │
│  ├── inspect.py   (Dataset Analysis)                   │
│  ├── doctor.py    (System Diagnostics)                 │
│  └── think.py     (AI Idea Analysis)                   │
├─────────────────────────────────────────────────────────┤
│  Core Layer                                             │
│  ├── templates.py   (Project Templates)                │
│  ├── analyzers.py   (Dataset Analyzers)                │
│  ├── diagnostics.py (System Checks)                    │
│  └── ai_advisor.py  (AI Recommendations)               │
├─────────────────────────────────────────────────────────┤
│  Utils Layer                                            │
│  ├── console.py  (Rich Console Utilities)              │
│  └── helpers.py  (Common Functions)                    │
└─────────────────────────────────────────────────────────┘
```

### Module Breakdown

#### 1. **Commands** (`yctl/commands/`)
- **init.py**: Creates new AI projects with templates
- **inspect.py**: Analyzes datasets and provides insights
- **doctor.py**: Checks system health for AI development
- **think.py**: Analyzes AI ideas and provides recommendations

#### 2. **Core** (`yctl/core/`)
- **templates.py**: Project templates (NLP, CV, ML, Research)
- **analyzers.py**: Dataset analysis engine
- **diagnostics.py**: System diagnostic checks
- **ai_advisor.py**: AI recommendation engine

#### 3. **Utils** (`yctl/utils/`)
- **console.py**: Rich console utilities for beautiful output
- **helpers.py**: Common utility functions

---

## 📚 Commands Reference

### 1. `yctl init`

**Purpose**: Initialize a new AI project

**Syntax**:
```bash
yctl init <project_type> <project_name> [OPTIONS]
```

**Project Types**:
- `nlp` - Natural Language Processing
- `cv` - Computer Vision
- `ml` - Machine Learning
- `research` - Research Projects

**Options**:
- `--skip-venv` - Skip virtual environment creation

**What it creates**:
- Complete directory structure
- Virtual environment (unless skipped)
- requirements.txt with relevant packages
- README.md with documentation
- .gitignore
- Starter code and configuration files

**Example**:
```bash
yctl init nlp sentiment-analyzer
```

---

### 2. `yctl inspect`

**Purpose**: Inspect and analyze datasets

**Syntax**:
```bash
yctl inspect <dataset_path> [OPTIONS]
```

**Supported Formats**:
- CSV
- Excel (.xlsx, .xls)
- JSON
- Parquet

**Options**:
- `--sample` - Show sample rows

**What it provides**:
- Dataset overview (rows, columns, memory)
- Column information and types
- Missing values analysis
- Numeric statistics (mean, std, min, max, quartiles)
- Categorical statistics (unique values, frequencies)
- Data quality issues
- Preprocessing suggestions
- Model recommendations

**Example**:
```bash
yctl inspect data/train.csv
```

---

### 3. `yctl doctor`

**Purpose**: Check system health for AI development

**Syntax**:
```bash
yctl doctor
```

**What it checks**:
- Python version (3.10+ recommended)
- pip installation and version
- venv module availability
- GPU detection (NVIDIA)
- CUDA availability and version
- PyTorch CUDA support
- Common development tools (git, docker, tmux, htop)

**What it provides**:
- Status for each component (OK, WARNING, ERROR)
- Detailed information
- Fix suggestions for issues

**Example**:
```bash
yctl doctor
```

---

### 4. `yctl think`

**Purpose**: Analyze AI project ideas

**Syntax**:
```bash
yctl think "<your_idea>"
```

**What it provides**:
- Feasibility assessment (High/Medium/Low)
- Complexity rating (Beginner/Intermediate/Advanced/Research)
- Step-by-step roadmap
- Suggested datasets with sources
- Model architectures
- Required tools and libraries
- Potential challenges
- Learning resources

**Example**:
```bash
yctl think "sentiment analysis for customer reviews"
```

---

## 📁 File Structure

### Complete Project Layout

```
yctl-cli/
├── yctl/                           # Main package
│   ├── __init__.py                # Package metadata
│   ├── main.py                    # CLI entry point (Typer app)
│   ├── commands/                  # Command implementations
│   │   ├── __init__.py
│   │   ├── init.py               # Project initialization
│   │   ├── inspect.py            # Dataset inspection
│   │   ├── doctor.py             # System diagnostics
│   │   └── think.py              # AI idea analysis
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── templates.py          # Project templates
│   │   ├── analyzers.py          # Dataset analyzers
│   │   ├── diagnostics.py        # System diagnostics
│   │   └── ai_advisor.py         # AI recommendations
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── console.py            # Rich console utilities
│       └── helpers.py            # Helper functions
├── examples/                      # Example scripts
│   └── demo.sh                   # Demo script
├── setup.py                       # Setup configuration
├── pyproject.toml                # Modern Python packaging
├── requirements.txt              # Dependencies
├── LICENSE                       # MIT License
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
├── EXAMPLES.md                   # Usage examples
├── PROJECT_SUMMARY.md            # Project summary
└── install_and_verify.sh         # Installation script
```

### Generated Project Structure (Example: NLP)

```
sentiment-analyzer/
├── data/
│   ├── raw/                      # Raw datasets
│   │   └── .gitkeep
│   └── processed/                # Preprocessed data
│       └── .gitkeep
├── notebooks/                    # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── models/                   # Model architectures
│   │   └── __init__.py
│   ├── preprocessing/            # Data preprocessing
│   │   └── __init__.py
│   ├── utils/                    # Utility functions
│   │   └── __init__.py
│   └── train.py                  # Main training script
├── tests/                        # Unit tests
│   └── __init__.py
├── configs/                      # Configuration files
│   └── config.yaml
├── outputs/
│   ├── models/                   # Saved models
│   │   └── .gitkeep
│   └── logs/                     # Training logs
│       └── .gitkeep
├── venv/                         # Virtual environment
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

---

## 🧪 Testing

### Manual Testing

```bash
# 1. Test installation
pip install -e .
yctl --help

# 2. Test doctor command
yctl doctor

# 3. Test inspect command
cat > /tmp/test.csv << EOF
id,name,value
1,Alice,100
2,Bob,200
EOF
yctl inspect /tmp/test.csv

# 4. Test think command
yctl think "sentiment analysis"

# 5. Test init command
yctl init nlp test-project --skip-venv
ls -la test-project/
rm -rf test-project
```

### Automated Testing

```bash
# Run verification script
./install_and_verify.sh
```

### Unit Testing (Future)

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=yctl tests/
```

---

## 💡 Usage Examples

### Example 1: Complete NLP Workflow

```bash
# Step 1: Check system
yctl doctor

# Step 2: Analyze idea
yctl think "sentiment analysis for movie reviews"

# Step 3: Create project
yctl init nlp movie-sentiment

# Step 4: Setup
cd movie-sentiment
source venv/bin/activate
pip install -r requirements.txt

# Step 5: Inspect data
yctl inspect ../data/reviews.csv

# Step 6: Train
python src/train.py --data data/processed/train.csv --epochs 10
```

### Example 2: Quick Dataset Analysis

```bash
# Inspect multiple datasets
yctl inspect data/train.csv
yctl inspect data/test.csv
yctl inspect data/validation.csv
```

### Example 3: System Troubleshooting

```bash
# Check system health
yctl doctor

# If GPU not detected, check drivers
nvidia-smi

# If CUDA not available, reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔧 Extending yctl

### Adding a New Project Type

Edit `yctl/core/templates.py`:

```python
class CustomTemplate(ProjectTemplate):
    """Template for custom projects."""
    
    def create_structure(self) -> None:
        # Define directory structure
        dirs = [
            "data/raw",
            "src/custom",
            # ... more directories
        ]
        for dir_path in dirs:
            (self.project_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_requirements(self) -> str:
        return """# Custom requirements
custom-package>=1.0.0
"""
    
    def get_readme(self) -> str:
        return f"""# {self.project_name}
Custom project documentation
"""

# Register the template
TEMPLATES["custom"] = CustomTemplate
```

### Adding a New Command

1. Create `yctl/commands/new_command.py`:

```python
import typer
from yctl.utils import console, print_header

def new_command(arg: str) -> None:
    """New command description."""
    print_header("New Command")
    console.print(f"Processing: {arg}")
    # Your logic here
```

2. Register in `yctl/main.py`:

```python
from yctl.commands import new_command

@app.command("new")
def new(arg: str = typer.Argument(...)):
    """New command."""
    new_command(arg)
```

### Customizing Output

Edit `yctl/utils/console.py` to customize Rich output:

```python
# Add new output function
def print_custom(text: str) -> None:
    """Print custom styled message."""
    console.print(f"[bold blue]{text}[/bold blue]")
```

---

## 📖 Documentation Files

- **README.md**: Main documentation with features and usage
- **QUICKSTART.md**: Quick start guide for new users
- **EXAMPLES.md**: Comprehensive usage examples
- **PROJECT_SUMMARY.md**: Project overview and architecture
- **This file**: Complete implementation guide

---

## 🎯 Best Practices

1. **Always activate virtual environment** before working on projects
2. **Use `yctl doctor`** before starting new projects
3. **Inspect datasets** before training to understand data quality
4. **Track experiments** with wandb or tensorboard
5. **Version control** your code and configurations
6. **Document experiments** as you go
7. **Use configuration files** for hyperparameters
8. **Write tests** for critical components

---

## 🤝 Contributing

To contribute to yctl:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- Typer - CLI framework
- Rich - Terminal output
- Pandas - Data analysis
- NumPy - Numerical computing
- Scikit-learn - ML utilities

---

**For more information, see:**
- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start
- [EXAMPLES.md](EXAMPLES.md) - Usage examples
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview
