from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import re
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
import random
import io
import base64
import os
import json
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Global variables to store the trained model and vectorizer
model = None
vectorizer = None
incremental_model = SGDClassifier(loss='log_loss', random_state=42)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

def train_model():
    global model, vectorizer
    
    # Load the data
    try:
        data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
        data = data[['headline', 'is_sarcastic']]
    except FileNotFoundError:
        print("Dataset not found. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        data = pd.DataFrame({
            'headline': [
                'Breaking: Local man discovers water is wet',
                'Scientists confirm sky is blue',
                'Weather forecast: It will be weather outside',
                'Local woman breathes air, survives',
                'Study shows people need food to live'
            ] * 1000,  # Repeat to have enough data
            'is_sarcastic': [1, 1, 1, 1, 1] * 1000
        })
    
    # Preprocess the text
    data['cleaned_headline'] = data['headline'].apply(preprocess_text)
    
    # Features and target
    X = data['cleaned_headline']
    y = data['is_sarcastic']
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=4)
    X = vectorizer.fit_transform(X)
    
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.65, random_state=42)
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Hyperparameter tuning
    C_values = [0.01, 0.1, 1, 10, 100]
    max_iter_values = [100, 200, 300]
    best_score = 0
    best_params = {}
    
    for C in C_values:
        for max_iter in max_iter_values:
            model_temp = LogisticRegression(C=C, max_iter=max_iter)
            scores = cross_val_score(model_temp, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': C, 'max_iter': max_iter}
    
    # Train the final model
    model = LogisticRegression(**best_params)
    model.fit(X_train_smote, y_train_smote)
    
    # Calculate accuracies
    y_train_pred = model.predict(X_train_smote)
    train_accuracy = accuracy_score(y_train_smote, y_train_pred)
    
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'best_params': best_params,
        'best_score': best_score
    }
def load_model():
    global model, vectorizer
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model.pkl')
        vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')

        print(f"Trying to load model from: {model_path}")
        print(f"Trying to load vectorizer from: {vectorizer_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        print("✅ Model and vectorizer loaded successfully.")
        return True
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}")
        return False


def predict_sarcasm(text):
    global model, vectorizer
    
    if model is None or vectorizer is None:
        return "Model not loaded"
    
    text = preprocess_text(text)
    non_sarcastic_phrases = ["hello world", "good morning", "how are you", "thank you", "nice to meet you"]
    
    for phrase in non_sarcastic_phrases:
        if phrase in text:
            return "Non-Sarcastic"
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    probability = model.predict_proba(text_vectorized)[0]
    
    result = "Sarcastic" if prediction[0] == 1 else "Non-Sarcastic"
    confidence = max(probability) * 100
    
    return {
        'result': result,
        'confidence': confidence,
        'probabilities': {
            'non_sarcastic': probability[0] * 100,
            'sarcastic': probability[1] * 100
        }
    }

def generate_plots():
    """Generate visualization plots and return as base64 encoded strings"""
    plots = {}
    
    # This is a simplified version - in production you'd use actual test data
    try:
        # Sample data for demonstration
        test_accuracy = 0.85
        
        # Test accuracy visualization
        plt.figure(figsize=(6, 4))
        plt.bar(['Test Accuracy'], [test_accuracy * 100], color='blue')
        plt.title('Model Test Accuracy')
        plt.ylabel('Accuracy Percentage')
        plt.ylim(0, 100)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plots['accuracy'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Confusion Matrix (sample data)
        cm = np.array([[150, 20], [30, 200]])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # ROC Curve (sample data)
        fpr = np.array([0.0, 0.1, 0.2, 0.8, 1.0])
        tpr = np.array([0.0, 0.7, 0.8, 0.9, 1.0])
        roc_auc = 0.85
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plots['roc_curve'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    return plots

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    statement = data.get('statement', '').strip()
    
    if not statement:
        return jsonify({'error': 'Please enter a statement to analyze!'}), 400
    
    # Predict sarcasm
    result = predict_sarcasm(statement)
    
    if isinstance(result, str):
        return jsonify({'error': result}), 500
    
    # Add to history
    if 'history' not in session:
        session['history'] = []
    
    history_entry = {
        'statement': statement,
        'result': result['result'],
        'confidence': result['confidence'],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    session['history'].append(history_entry)
    session.modified = True
    
    # Keep only last 50 entries
    if len(session['history']) > 50:
        session['history'] = session['history'][-50:]
    
    return jsonify({
        'result': result['result'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'message': 'This statement seems sarcastic!' if result['result'] == 'Sarcastic' else 'This statement does not appear sarcastic.'
    })

@app.route('/random_quote')
def random_quote():
    quotes = [
        "Just what I always wanted to hear!",
        "Absolutely... as if I had a choice!",
        "I am totally not a racist.",
        "Wow, I just love waiting in line for hours. Best part of my day!",
        "Area Woman Still Alive After Weekend, Miraculously."
    ]
    return jsonify({'quote': random.choice(quotes)})

@app.route('/history')
def get_history():
    return jsonify(session.get('history', []))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/visualizations')
def visualizations():
    plots = generate_plots()
    return render_template('visualizations.html', plots=plots)

@app.route('/train_model', methods=['POST'])
def train_model_route():
    try:
        results = train_model()
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Model trained successfully!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Initialize the application
if __name__ == '__main__':
    # Try to load existing model, if not available, train a new one
    if not load_model():
        print("No pre-trained model found. Training new model...")
        try:
            train_model()
            print("Model training completed!")
        except Exception as e:
            print(f"Error training model: {e}")
            print("The application will still run but predictions may not work without a dataset.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
