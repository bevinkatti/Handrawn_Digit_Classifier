// static/js/script.js - Smooth Drawing Version
class DigitClassifier {
    constructor() {
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        
        this.initializeCanvas();
        this.setupEventListeners();
        this.generateProbabilityBars();
    }

    initializeCanvas() {
        // Set canvas background to white
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.lineWidth = 20;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = 'black';
        this.ctx.globalCompositeOperation = 'source-over';
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Button events
        document.getElementById('predictBtn').addEventListener('click', this.predictDigit.bind(this));
        document.getElementById('clearBtn').addEventListener('click', this.clearCanvas.bind(this));
    }

    getCanvasCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        let clientX, clientY;
        
        if (e.type.includes('touch')) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        e.preventDefault();
        this.isDrawing = true;
        const pos = this.getCanvasCoordinates(e);
        
        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
        
        // Draw a dot for single clicks
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
    }

    draw(e) {
        if (!this.isDrawing) return;
        e.preventDefault();
        
        const pos = this.getCanvasCoordinates(e);
        
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.beginPath(); // Reset path for next drawing
        }
    }

    handleTouchStart(e) {
        e.preventDefault();
        this.startDrawing(e);
    }

    handleTouchMove(e) {
        e.preventDefault();
        this.draw(e);
    }

    clearCanvas() {
        // Clear the canvas
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Reset drawing state
        this.ctx.beginPath();
        this.isDrawing = false;
        
        this.resetResults();
    }

    resetResults() {
        document.getElementById('predictedDigit').textContent = '-';
        document.getElementById('confidence').textContent = 'Confidence: -';
        this.updateProbabilityBars([]);
    }

    async predictDigit() {
        const predictBtn = document.getElementById('predictBtn');
        const originalText = predictBtn.textContent;
        
        try {
            predictBtn.textContent = 'Predicting...';
            predictBtn.disabled = true;
            
            console.log('🎯 Starting prediction...');
            
            // Convert canvas to base64
            const imageData = this.canvas.toDataURL('image/png');
            console.log('📷 Canvas converted to base64');
            
            // Send to server for prediction
            console.log('📤 Sending request to /predict...');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            console.log('📥 Response received, status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('📊 Prediction result:', result);

            if (result.success) {
                this.displayResults(result);
                console.log('✅ Prediction successful:', result.prediction);
            } else {
                throw new Error(result.error || 'Unknown prediction error');
            }

        } catch (error) {
            console.error('❌ Prediction error:', error);
            alert('Error making prediction: ' + error.message);
        } finally {
            predictBtn.textContent = originalText;
            predictBtn.disabled = false;
        }
    }

    displayResults(result) {
        document.getElementById('predictedDigit').textContent = result.prediction;
        document.getElementById('confidence').textContent = 
            `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
        
        this.updateProbabilityBars(result.probabilities);
    }

    generateProbabilityBars() {
        const container = document.getElementById('probabilityBars');
        container.innerHTML = '';
        
        for (let i = 0; i < 10; i++) {
            const barHTML = `
                <div class="probability-bar">
                    <div class="probability-label">
                        <span class="digit">${i}</span>
                        <span class="probability-value" id="probValue${i}">0%</span>
                    </div>
                    <div class="probability-track">
                        <div class="probability-fill" id="probFill${i}" style="width: 0%"></div>
                    </div>
                </div>
            `;
            container.innerHTML += barHTML;
        }
    }

    updateProbabilityBars(probabilities) {
        for (let i = 0; i < 10; i++) {
            const percentage = probabilities && probabilities[i] ? (probabilities[i] * 100).toFixed(1) : 0;
            document.getElementById(`probValue${i}`).textContent = `${percentage}%`;
            document.getElementById(`probFill${i}`).style.width = `${percentage}%`;
        }
    }
}

// Initialize with error handling
document.addEventListener('DOMContentLoaded', () => {
    try {
        new DigitClassifier();
        console.log('✅ Digit classifier initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize digit classifier:', error);
    }
});