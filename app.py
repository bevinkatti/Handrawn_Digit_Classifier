from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
model = None

def load_model():
    global model
    if model is None:
        model_path = 'model/mnist_model.h5'
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print("✅ Model loaded successfully!")
                print(f"📊 TensorFlow version: {tf.__version__}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        else:
            print(f"❌ Model file not found at {model_path}")
            return False
    return True

def preprocess_image(image_data):
    try:
        print("🖼️ Starting image preprocessing...")
        
        # Convert base64 image data to PIL Image
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        print(f"📐 Image decoded, size: {image.size}")
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize((28, 28))
        print("🎨 Image converted to grayscale and resized")
        
        # Convert to numpy array and invert colors (MNIST has white digits on black background)
        image_array = np.array(image)
        print(f"📊 Original image range: {np.min(image_array)} to {np.max(image_array)}")
        
        image_array = 255 - image_array  # Invert colors
        print(f"🔄 After inversion range: {np.min(image_array)} to {np.max(image_array)}")
        
        # Normalize and reshape for model
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        
        print(f"✅ Final preprocessed image shape: {image_array.shape}")
        print(f"📈 Final image range: {np.min(image_array):.3f} to {np.max(image_array):.3f}")
        
        return image_array
    
    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🎯 Received prediction request")
        
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            })
        
        print(f"📨 Image data received, length: {len(data['image'])}")
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        print("🧠 Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities
        probabilities = predictions[0].tolist()
        
        print(f"✅ Prediction: {predicted_digit}, Confidence: {confidence:.4f}")
        
        return jsonify({
            'success': True,
            'prediction': int(predicted_digit),
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/test', methods=['GET'])
def test_model():
    """Test endpoint to verify model is working"""
    try:
        print("🧪 Testing model...")
        
        # Create a test image (digit 5 pattern)
        test_image = np.zeros((1, 28, 28, 1), dtype='float32')
        test_image[0, 10:18, 10:18, 0] = 0.8  # Simple square pattern
        
        predictions = model.predict(test_image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        print(f"🧪 Test prediction: {predicted_digit}, Confidence: {confidence:.4f}")
        
        return jsonify({
            'success': True,
            'message': 'Model is working!',
            'test_prediction': int(predicted_digit),
            'test_confidence': confidence
        })
    
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__
    })

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    if load_model():
        print("🌐 Starting web server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("💥 Failed to load model. Exiting.")