import numpy as np
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path, seq_length=340):
    # Load the model
    model = load_model(model_path)
    
    # Load the vectorizer
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

# Preprocess function to convert text into vectorized and padded format
def preprocess_text(text, vectorizer, seq_length):
    text_vec = [vectorizer.get(word, 0) for word in text.split()]
    padded_vec = pad_sequences([text_vec], maxlen=seq_length, padding='post', truncating='post')
    return np.array(padded_vec)

# Prediction function to determine if the input text is AI-generated
def predict_ai_generated(input_text, model, vectorizer, seq_length):
    processed_text = preprocess_text(input_text, vectorizer, seq_length)
    prediction = model.predict(processed_text)
    is_ai_generated = prediction[0][0] >= 0.5  # Assuming 0.5 as the threshold
    return "AI Generated" if is_ai_generated else "Human"

# Load the model and vectorizer when the app starts
model, vectorizer = load_model_and_vectorizer('text_classifier_model.h5', 'vectorizer.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        result = predict_ai_generated(input_text, model, vectorizer, seq_length=340)
        return render_template('index.html', result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
