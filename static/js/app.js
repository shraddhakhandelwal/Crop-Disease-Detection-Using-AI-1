// Crop Disease Detection - Main JavaScript

class CropDiseaseDetector {
    constructor() {
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.previewSection = document.getElementById('previewSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.loading = document.getElementById('loading');
        this.originalImage = document.getElementById('originalImage');
        this.gradcamImage = document.getElementById('gradcamImage');
        this.predictBtn = document.getElementById('predictBtn');
        this.resetBtn = document.getElementById('resetBtn');
        
        this.selectedFile = null;
        this.apiUrl = window.location.origin;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIHealth();
    }

    setupEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            this.handleFileSelect(file);
        });

        // Predict button
        this.predictBtn.addEventListener('click', () => {
            this.predictDisease();
        });

        // Reset button
        this.resetBtn.addEventListener('click', () => {
            this.reset();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showAlert('Please select a valid image file (JPEG, PNG)', 'error');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showAlert('File size must be less than 10MB', 'error');
            return;
        }

        this.selectedFile = file;
        this.showPreview(file);
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImage.src = e.target.result;
            this.previewSection.classList.add('active');
            this.predictBtn.disabled = false;
            
            // Hide results from previous prediction
            this.resultsSection.classList.remove('active');
            this.gradcamImage.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async predictDisease() {
        if (!this.selectedFile) {
            this.showAlert('Please select an image first', 'warning');
            return;
        }

        // Show loading
        this.loading.classList.add('active');
        this.predictBtn.disabled = true;
        this.resultsSection.classList.remove('active');

        const formData = new FormData();
        formData.append('image', this.selectedFile);

        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data);
                
                // Get GradCAM visualization if available
                if (data.gradcam_available) {
                    this.getGradCAM();
                }
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showAlert(`Error: ${error.message}`, 'error');
        } finally {
            this.loading.classList.remove('active');
            this.predictBtn.disabled = false;
        }
    }

    async getGradCAM() {
        const formData = new FormData();
        formData.append('image', this.selectedFile);

        try {
            const response = await fetch(`${this.apiUrl}/gradcam`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                this.gradcamImage.src = imageUrl;
                this.gradcamImage.style.display = 'block';
            }
        } catch (error) {
            console.error('GradCAM error:', error);
        }
    }

    displayResults(data) {
        const prediction = data.prediction;
        
        // Main prediction
        document.getElementById('diseaseName').textContent = prediction.disease;
        document.getElementById('confidencePercentage').textContent = 
            `${(prediction.confidence * 100).toFixed(2)}%`;
        
        // Update confidence badge color
        const badge = document.getElementById('confidenceBadge');
        badge.className = 'confidence-badge';
        if (prediction.confidence >= 0.8) {
            badge.classList.add('success');
        } else if (prediction.confidence >= 0.5) {
            badge.classList.add('warning');
        } else {
            badge.classList.add('danger');
        }

        // Confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.style.width = `${prediction.confidence * 100}%`;
        confidenceBar.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

        // Top predictions
        if (data.top_predictions && data.top_predictions.length > 0) {
            const topPredictionsContainer = document.getElementById('topPredictions');
            topPredictionsContainer.innerHTML = '';
            
            data.top_predictions.forEach((pred, index) => {
                const predItem = document.createElement('div');
                predItem.className = 'prediction-item fade-in';
                predItem.style.animationDelay = `${index * 0.1}s`;
                predItem.innerHTML = `
                    <span class="prediction-name">${index + 1}. ${pred.disease}</span>
                    <span class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
                `;
                topPredictionsContainer.appendChild(predItem);
            });
        }

        // Show results section
        this.resultsSection.classList.add('active');
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                console.log('API is healthy');
                this.showAlert('System ready! Upload an image to detect crop diseases.', 'success');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            this.showAlert('Warning: API connection issue. Please ensure the server is running.', 'warning');
        }
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        document.querySelectorAll('.alert').forEach(alert => {
            alert.remove();
        });

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} active fade-in`;
        alert.textContent = message;
        
        const uploadSection = document.querySelector('.upload-section');
        uploadSection.insertBefore(alert, uploadSection.firstChild);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            alert.classList.remove('active');
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    }

    reset() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.previewSection.classList.remove('active');
        this.resultsSection.classList.remove('active');
        this.loading.classList.remove('active');
        this.predictBtn.disabled = true;
        
        // Clear images
        this.originalImage.src = '';
        this.gradcamImage.src = '';
        this.gradcamImage.style.display = 'none';
        
        // Remove alerts
        document.querySelectorAll('.alert').forEach(alert => {
            alert.remove();
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new CropDiseaseDetector();
});

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});
