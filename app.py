import os
import json
import re # Import the regular expressions library
import fitz  # PyMuPDF
import google.generativeai as genai # Import Google's library
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- NEW: Configure the Google Gemini Client ---
try:
    # It automatically reads the GOOGLE_API_KEY from your environment variables
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Google Gemini client: {e}")

def generate_quiz_from_text(text):
    """
    Sends extracted text to the Google Gemini API to generate quiz questions.
    """
    # --- NEW: Initialize the Gemini Pro model ---
    model = genai.GenerativeModel('gemini-pro')

    num_mcq = 5
    num_tf = 5

    # This prompt is crucial and has been slightly optimized for Gemini.
    prompt = f"""
    Based on the following text, please generate a quiz. The quiz must contain exactly {num_mcq} multiple-choice questions
    and {num_tf} true/false questions.

    The text is as follows:
    ---
    {text[:8000]} 
    ---

    You MUST format the output as a single, valid JSON object with two keys: "multiple_choice" and "true_false".
    - The value for "multiple_choice" must be an array of objects, each with "question", "options" (an array of exactly 4 strings), and "answer".
    - The value for "true_false" must be an array of objects, each with "question" and "answer" (a string, either "True" or "False").
    Do not include any text, notes, or markdown formatting before or after the JSON object.
    """

    try:
        # --- NEW: Call the Gemini API ---
        response = model.generate_content(prompt)
        
        # --- NEW: Clean up the response to ensure it's valid JSON ---
        # Gemini sometimes wraps the JSON in ```json ... ```, so we extract it.
        response_text = response.text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            clean_json = response_text

        return json.loads(clean_json)

    except Exception as e:
        print(f"An unexpected error occurred during AI generation: {e}")
        print(f"Raw response from AI was: {response.text if 'response' in locals() else 'No response'}")
        raise ValueError("Failed to get a valid JSON response from the AI model.")


# The rest of the file remains the same...
@app.route('/')
def index():
    return "<h1>Quiz Generator Backend (Gemini Version)</h1><p>The server is running.</p>"

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_endpoint():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file was provided."}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No file was selected."}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    try:
        pdf_bytes = file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        
        if not text.strip():
            return jsonify({"error": "Could not extract text from the PDF."}), 400

        quiz_data = generate_quiz_from_text(text)
        
        if "multiple_choice" not in quiz_data or "true_false" not in quiz_data:
            raise ValueError("The generated quiz from the AI is missing required sections.")

        return jsonify(quiz_data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
