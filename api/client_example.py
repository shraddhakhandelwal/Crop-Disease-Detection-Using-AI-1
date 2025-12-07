"""
Example client for testing the API.

Usage:
    python api/client_example.py --image path/to/image.jpg
"""
import requests
import argparse
from pathlib import Path
import json


def predict_single_image(api_url: str, image_path: str, top_k: int = 3):
    """Send single image to API for prediction."""
    
    endpoint = f"{api_url}/predict"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'top_k': top_k}
        
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nPrediction for: {image_path}")
        print(f"  Disease: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Confident: {result['is_confident']}")
        
        if 'top_predictions' in result:
            print(f"\n  Top {top_k} predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"    {i}. {pred['class']}: {pred['confidence']:.2%}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def get_classes(api_url: str):
    """Get list of available classes."""
    
    endpoint = f"{api_url}/classes"
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nAvailable classes ({result['num_classes']}):")
        for i, class_name in enumerate(result['classes'], 1):
            print(f"  {i}. {class_name}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None


def health_check(api_url: str):
    """Check API health."""
    
    endpoint = f"{api_url}/health"
    response = requests.get(endpoint)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nAPI Health:")
        print(f"  Status: {result['status']}")
        print(f"  Model Loaded: {result['model_loaded']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test crop disease detection API')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000',
                       help='API base URL')
    parser.add_argument('--image', type=str,
                       help='Path to image for prediction')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions')
    parser.add_argument('--check_health', action='store_true',
                       help='Check API health')
    parser.add_argument('--list_classes', action='store_true',
                       help='List available classes')
    
    args = parser.parse_args()
    
    # Health check
    if args.check_health:
        health_check(args.api_url)
    
    # List classes
    if args.list_classes:
        get_classes(args.api_url)
    
    # Predict
    if args.image:
        predict_single_image(args.api_url, args.image, args.top_k)
    
    # If no arguments, show help
    if not (args.check_health or args.list_classes or args.image):
        parser.print_help()
