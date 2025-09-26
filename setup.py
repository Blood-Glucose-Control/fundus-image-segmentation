# Setup and installation script for fundus image segmentation

import os
import subprocess
import sys
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, CUDA check will be performed after installation")
        return False

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing requirements"):
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "dataset/train/images",
        "dataset/train/masks", 
        "dataset/val/images",
        "dataset/val/masks",
        "dataset/test/images",
        "dataset/test/masks",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/results"
    ]
    
    print("\nCreating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# MacOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini

# Project specific
outputs/
dataset/
logs/
checkpoints/
*.pth
*.pkl
*.h5
*.hdf5

# Jupyter Notebooks
.ipynb_checkpoints

# TensorBoard
runs/

# Large files
*.zip
*.tar.gz
*.rar
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    print("✅ Created .gitignore file")
    return True

def verify_installation():
    """Verify that the installation was successful"""
    print("\nVerifying installation...")
    
    try:
        # Test imports
        import torch
        import torchvision
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import albumentations
        from tqdm import tqdm
        
        # Test our modules
        sys.path.append('src')
        from models import UNet
        from data import FundusDataset
        from training import Trainer
        from evaluation import ModelEvaluator
        from utils import visualize_segmentation
        
        print("✅ All required packages imported successfully")
        
        # Test model creation
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your dataset to the dataset/ directory")
    print("   - Place images in dataset/train/images and dataset/val/images")
    print("   - Place masks in dataset/train/masks and dataset/val/masks")
    print()
    print("2. Train your model:")
    print("   python scripts/train.py --data_dir dataset --epochs 50")
    print()
    print("3. Run inference:")
    print("   python scripts/inference.py --model_path outputs/checkpoints/best_model.pth --image_path your_image.jpg")
    print()
    print("4. Alternatively, use the Google Colab notebook:")
    print("   notebooks/Fundus_Image_Segmentation_Colab.ipynb")
    print()
    print("📖 Check README.md for detailed documentation")
    print("🐛 Report issues at: https://github.com/Blood-Glucose-Control/fundus-image-segmentation/issues")
    print("="*60)

def main():
    """Main setup function"""
    print("🔧 Fundus Image Segmentation - Setup Script")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Create .gitignore
    create_gitignore()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()