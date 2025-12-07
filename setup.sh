#!/bin/bash
# Setup script for Crop Disease Detection project

echo "Setting up Crop Disease Detection project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{raw,processed,samples}
mkdir -p models/{checkpoints,saved_models}
mkdir -p logs
mkdir -p results

# Create .gitkeep files to preserve directory structure
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/samples/.gitkeep
touch models/checkpoints/.gitkeep
touch models/saved_models/.gitkeep

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download dataset and place in data/raw/"
echo "3. Update config/config.yaml with your settings"
echo "4. Run training: python train.py --data_dir data/raw"
echo ""
echo "For PlantVillage dataset, you can download from:"
echo "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
echo ""
