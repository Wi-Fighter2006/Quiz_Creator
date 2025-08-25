import os
import json
import re
import fitz  # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Configure the Google Gemini Client
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Google Gemini client: {e}")

def generate_quiz_from_text(text):
    model = genai.GenerativeModel('gemini-1.0-pro')

    # --- NEW: Define less strict safety settings ---
    # This is often necessary for processing diverse academic texts.
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    Based on the following text, please generate a quiz. The quiz must contain exactly 5 multiple-choice questions
    and 5 true/false questions.

    The text is as follows:
    ---
    {text[:8000]} 
    ---

    You MUST format the output as a single, valid JSON object with two keys: "multiple_choice" and "true_false".
    - The value for "multiple_choice" must be an array of objects, each with "question", "options" (an array of exactly 4 strings), and "answer".
    - The value for "true_false" must be an array of objects, each with "question" and "answer" (a string, either "True" or "False").
    Do not include any text, notes, or markdown formatting before or after the JSON object.
    """
    
    response_text = ""
    try:
        # --- MODIFIED: Pass the safety_settings to the API call ---
        response = model.generate_content(prompt, safety_settings=safety_settings)
        response_text = response.text
        
        # Clean up the response to ensure it's valid JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            # If no markdown block is found, assume the whole text is the JSON
            clean_json = response_text

        return json.loads(clean_json)

    except Exception as e:
        # --- IMPROVED LOGGING ---
        # This will show us exactly what the AI returned if something goes wrong.
        print(f"An error occurred during AI generation: {e}")
        print(f"--- RAW RESPONSE FROM AI WAS ---")
        print(response_text)
        print(f"--- END OF RAW RESPONSE ---")
        raise ValueError("Failed to get a valid JSON response from the AI model.")


# --- No changes below this line ---

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
