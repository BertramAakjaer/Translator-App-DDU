from flask import Flask, request, jsonify
from flask_cors import CORS
import os, webbrowser, tempfile, re


from server_side.translator import translate

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

PORT = 8000  # Choose a suitable port

# List to store translation history
translation_history = []

def open_website():
    # Define the path to the HTML file
    html_file_path = 'index.html'
    abs_path = os.path.abspath(html_file_path)
    webbrowser.open(f'file://{abs_path}')

def translate_sentences(file_to_translate):
    sentences = []
    
    with open(file_to_translate, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    
    translated_sentences = []
    for sentence in sentences:
        if not re.search('[a-zA-Z]', sentence):
            translated_sentences.append(sentence)
        else:
            translated_sentence = translate(sentence, False)
            translated_sentences.append(translated_sentence)
    
    combined_translated_sentences = '\n'.join(translated_sentences)
    
    return combined_translated_sentences

@app.route('/process_string', methods=['POST'])
def process_string():
    data = request.get_json()
    input_string = data.get('input_string', '')

    # Process the input_string here
    output_string = translate(input_string)

    # Add to history
    translation_history.append({'input': input_string, 'output': output_string})

    return jsonify({'output_string': output_string})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        return jsonify({'output_string': translate_sentences(file_path)})

@app.route('/get_history', methods=['GET'])
def fetch_history():  # Renamed function to avoid conflict
    reversed_history = list(reversed(translation_history))
    
    return jsonify(reversed_history)

if __name__ == '__main__':
    open_website()
    app.run(port=PORT)
