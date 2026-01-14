#!/bin/bash

# Installation and Verification Script for yctl
# This script installs yctl and runs basic verification tests

set -e  # Exit on error

echo "=========================================="
echo "yctl Installation and Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
echo ""

# Install yctl
echo "Installing yctl..."
pip install -e . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "yctl installed successfully"
else
    print_error "Failed to install yctl"
    exit 1
fi
echo ""

# Verify installation
echo "Verifying installation..."
if command -v yctl &> /dev/null; then
    print_success "yctl command is available"
else
    print_error "yctl command not found"
    exit 1
fi
echo ""

# Test help command
echo "Testing help command..."
yctl --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Help command works"
else
    print_error "Help command failed"
    exit 1
fi
echo ""

# Test doctor command
echo "Testing doctor command..."
yctl doctor > /tmp/yctl_doctor_output.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Doctor command works"
else
    print_warning "Doctor command completed with warnings (this is normal)"
fi
echo ""

# Create test dataset
echo "Creating test dataset..."
cat > /tmp/yctl_test_data.csv << 'EOF'
id,name,age,salary,department
1,Alice,25,50000,Engineering
2,Bob,30,60000,Engineering
3,Charlie,35,70000,Sales
4,David,28,55000,Marketing
5,Eve,32,65000,Engineering
EOF
print_success "Test dataset created"
echo ""

# Test inspect command
echo "Testing inspect command..."
yctl inspect /tmp/yctl_test_data.csv > /tmp/yctl_inspect_output.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Inspect command works"
else
    print_error "Inspect command failed"
    exit 1
fi
echo ""

# Test think command
echo "Testing think command..."
yctl think "sentiment analysis" > /tmp/yctl_think_output.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Think command works"
else
    print_error "Think command failed"
    exit 1
fi
echo ""

# Test init command
echo "Testing init command..."
cd /tmp
rm -rf yctl_test_project 2>/dev/null
yctl init nlp yctl_test_project --skip-venv > /tmp/yctl_init_output.txt 2>&1
if [ $? -eq 0 ]; then
    print_success "Init command works"
    
    # Verify project structure
    if [ -d "yctl_test_project" ]; then
        print_success "Project directory created"
        
        # Check for key files
        if [ -f "yctl_test_project/README.md" ]; then
            print_success "README.md created"
        fi
        
        if [ -f "yctl_test_project/requirements.txt" ]; then
            print_success "requirements.txt created"
        fi
        
        if [ -d "yctl_test_project/src" ]; then
            print_success "src/ directory created"
        fi
        
        # Cleanup
        rm -rf yctl_test_project
    fi
else
    print_error "Init command failed"
    exit 1
fi
echo ""

# Cleanup
rm -f /tmp/yctl_test_data.csv
rm -f /tmp/yctl_doctor_output.txt
rm -f /tmp/yctl_inspect_output.txt
rm -f /tmp/yctl_think_output.txt
rm -f /tmp/yctl_init_output.txt

echo "=========================================="
print_success "All tests passed!"
echo "=========================================="
echo ""
echo "yctl is ready to use!"
echo ""
echo "Quick start:"
echo "  1. yctl doctor              # Check system health"
echo "  2. yctl think \"your idea\"   # Analyze an AI idea"
echo "  3. yctl init <type> <name>  # Create a project"
echo "  4. yctl inspect <dataset>   # Inspect a dataset"
echo ""
echo "For more information:"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo "  - EXAMPLES.md"
echo ""
