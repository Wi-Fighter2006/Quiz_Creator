import os
import json
import fitz  # PyMuPDF
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) to allow the frontend to call the backend
CORS(app)

# Configure the OpenAI client
# It automatically reads the OPENAI_API_KEY from your environment variables on Render
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def generate_quiz_from_text(text):
    """
    Sends extracted text to the OpenAI API to generate quiz questions.
    """
    if not client:
        raise ConnectionError("OpenAI client is not initialized. Check your API key in Render.")

    num_mcq = 5
    num_tf = 5

    # This prompt is crucial. It clearly instructs the model on the task and JSON format.
    prompt = f"""
    Based on the following text, please generate a quiz. The quiz should contain {num_mcq} multiple-choice questions
    and {num_tf} true/false questions.

    The text is as follows:
    ---
    {text[:4000]}
    ---

    Please format the output as a single valid JSON object with two keys: "multiple_choice" and "true_false".
    - The value for "multiple_choice" must be an array of objects, each with "question", "options" (an array of 4 strings), and "answer" (the correct option string).
    - The value for "true_false" must be an array of objects, each with "question" and "answer" (a string, either "True" or "False").
    """

    try:
        completion = client.chat.completions.create(
            # Using the standard, reliable model for this task.
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to create quizzes for educators and output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        raise ConnectionAbortedError(f"Failed to get a response from the AI model: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from LLM response. Response was: {response_content}")
        raise ValueError("The AI model returned a response that was not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred during AI generation: {e}")
        raise

@app.route('/')
def index():
    """A simple route to test if the backend server is running."""
    return "<h1>Quiz Generator Backend</h1><p>The server is running. Use the /generate-quiz endpoint to create a quiz.</p>"

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_endpoint():
    """
    The main API endpoint to handle PDF upload and quiz generation.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file was provided in the request."}), 400

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
            return jsonify({"error": "Could not extract any text from the PDF. The file might be empty or image-based."}), 400

        quiz_data = generate_quiz_from_text(text)
        
        if "multiple_choice" not in quiz_data or "true_false" not in quiz_data:
            raise ValueError("The generated quiz from the AI is missing required sections.")

        return jsonify(quiz_data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Return a generic server error message to the user for security
        return jsonify({"error": f"An internal server error occurred. Please check the logs."}), 500

if __name__ == '__main__':
    # For local development only
    app.run(host='0.0.0.0', port=5000, debug=True)
