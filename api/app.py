"""
Flask API for crop disease detection.

Usage:
    python api/app.py

API Endpoints:
    GET / - Web interface
    POST /predict - Predict disease from uploaded image
    POST /gradcam - Get Grad-CAM visualization
    GET /health - Health check
    GET /classes - Get list of available classes
"""
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pathlib import Path
import json
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import DiseasePredictor
from src.inference.gradcam import GradCAM
from src.utils import load_config, setup_logger

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)

# Setup logger
logger = setup_logger(__name__)

# Load configuration
config = load_config('config/config.yaml')
api_config = config.get('api', {})

# Initialize predictor (will be loaded on first request)
predictor = None
class_names = None


def load_model():
    """Load model and class names."""
    global predictor, class_names
    
    if predictor is not None:
        return
    
    # Load class names
    class_names_path = Path('models/saved_models/class_names.json')
    if not class_names_path.exists():
        raise FileNotFoundError("Class names file not found. Please train a model first.")
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Find latest model
    model_dir = Path('models/saved_models')
    model_files = list(model_dir.glob('*.h5'))
    
    if not model_files:
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    # Use the most recent model
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Loading model from {model_path}")
    
    # Initialize predictor
    predictor = DiseasePredictor(
        model_path=str(model_path),
        class_names=class_names,
        confidence_threshold=config['inference'].get('confidence_threshold', 0.7)
    )
    
    logger.info("Model loaded successfully")


@app.route('/', methods=['GET'])
def index():
    """Serve web interface."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of disease classes."""
    try:
        load_model()
        return jsonify({
            'classes': class_names,
            'num_classes': len(class_names)
        })
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image."""
    try:
        # Load model if not loaded
        load_model()
        
        # Check if image in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        image = np.array(image.convert('RGB'))
        
        # Get parameters
        top_k = request.form.get('top_k', 3, type=int)
        
        # Make prediction
        result = predictor.predict(image, top_k=top_k)
        
        # Format response
        response = {
            'success': True,
            'prediction': {
                'disease': result['predicted_class'],
                'confidence': float(result['confidence']),
                'is_confident': result['is_confident']
            },
            'top_predictions': [
                {
                    'disease': pred['class'],
                    'confidence': float(pred['confidence'])
                }
                for pred in result['top_predictions']
            ],
            'gradcam_available': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict diseases for multiple images."""
    try:
        load_model()
        
        # Check if images in request
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        image_files = request.files.getlist('images')
        
        if not image_files:
            return jsonify({'error': 'No images provided'}), 400
        
        # Process images
        images = []
        filenames = []
        
        for image_file in image_files:
            if image_file.filename:
                image_bytes = image_file.read()
                image = Image.open(BytesIO(image_bytes))
                image = np.array(image.convert('RGB'))
                images.append(image)
                filenames.append(image_file.filename)
        
        if not images:
            return jsonify({'error': 'No valid images provided'}), 400
        
        # Make predictions
        results = predictor.predict_batch(images)
        
        # Format response
        response = {
            'success': True,
            'count': len(results),
            'predictions': [
                {
                    'filename': filename,
                    'predicted_class': result['predicted_class'],
                    'confidence': float(result['confidence']),
                    'is_confident': result['is_confident']
                }
                for filename, result in zip(filenames, results)
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/gradcam', methods=['POST'])
def gradcam():
    """Generate Grad-CAM visualization for uploaded image."""
    try:
        # Load model if not loaded
        load_model()
        
        # Check if image in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image.convert('RGB'))
        
        # Generate Grad-CAM
        try:
            gradcam_generator = GradCAM(predictor.model)
            heatmap = gradcam_generator.generate_heatmap(image_array)
            overlay = gradcam_generator.overlay_heatmap(image_array, heatmap)
            
            # Convert to bytes
            is_success, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            io_buf = BytesIO(buffer)
            
            return send_file(io_buf, mimetype='image/png')
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return jsonify({'error': 'Could not generate Grad-CAM visualization'}), 500
    
    except Exception as e:
        logger.error(f"Error in Grad-CAM endpoint: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.warning("Model will be loaded on first request")
    
    # Run app
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
