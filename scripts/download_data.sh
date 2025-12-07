#!/bin/bash
# Download PlantVillage dataset script

echo "PlantVillage Dataset Download Script"
echo "======================================"
echo ""
echo "This script will help you download the PlantVillage dataset."
echo ""
echo "Option 1: Manual Download"
echo "  1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
echo "  2. Download the dataset"
echo "  3. Extract to data/raw/"
echo ""
echo "Option 2: Kaggle API (requires Kaggle account and API key)"
echo "  1. Install Kaggle: pip install kaggle"
echo "  2. Setup API key: https://www.kaggle.com/docs/api"
echo "  3. Run: kaggle datasets download -d abdallahalidev/plantvillage-dataset"
echo "  4. Extract to data/raw/"
echo ""

read -p "Do you want to download using Kaggle API? (y/n): " choice

if [ "$choice" == "y" ]; then
    # Check if kaggle is installed
    if ! command -v kaggle &> /dev/null; then
        echo "Installing Kaggle..."
        pip install kaggle
    fi
    
    # Check if API key is configured
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo ""
        echo "Kaggle API key not found!"
        echo "Please follow these steps:"
        echo "1. Go to https://www.kaggle.com/account"
        echo "2. Scroll to 'API' section"
        echo "3. Click 'Create New Token'"
        echo "4. Move the downloaded kaggle.json to ~/.kaggle/"
        echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
    
    echo "Downloading dataset..."
    mkdir -p data/downloads
    cd data/downloads
    
    kaggle datasets download -d abdallahalidev/plantvillage-dataset
    
    echo "Extracting dataset..."
    unzip -q plantvillage-dataset.zip -d ../raw/
    
    echo "Cleaning up..."
    cd ../..
    
    echo ""
    echo "Dataset downloaded successfully to data/raw/"
    echo "You can now run: python train.py --data_dir data/raw"
else
    echo ""
    echo "Please download the dataset manually and extract to data/raw/"
fi
