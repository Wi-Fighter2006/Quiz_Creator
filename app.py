import os
import json
import fitz  # PyMuPDF
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the frontend

# Configure the OpenAI client
# The client automatically reads the OPENAI_API_KEY from the environment
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def generate_quiz_from_text(text):
    """
    Sends the extracted text to an LLM to generate quiz questions.
    """
    if not client:
        raise ConnectionError("OpenAI client is not initialized. Check your API key.")

    # You can adjust the number of questions here
    num_mcq = 5
    num_tf = 5

    # This is the prompt that instructs the LLM.
    # It is crucial to specify the exact JSON format you want.
    prompt = f"""
    Based on the following text, please generate a quiz. The quiz should contain {num_mcq} multiple-choice questions
    and {num_tf} true/false questions.

    The text is as follows:
    ---
    {text[:4000]}
    ---

    Please format the output as a single JSON object with two keys: "multiple_choice" and "true_false".
    - Each element in the "multiple_choice" array should be an object with "question", "options" (an array of 4 strings), and "answer" (the correct option string).
    - Each element in the "true_false" array should be an object with "question" and "answer" (either "True" or "False").

    Example of the required JSON format:
    {{
      "multiple_choice": [
        {{
          "question": "What is the capital of France?",
          "options": ["London", "Berlin", "Paris", "Madrid"],
          "answer": "Paris"
        }}
      ],
      "true_false": [
        {{
          "question": "The sky is blue.",
          "answer": "True"
        }}
      ]
    }}
    """

    try:
        completion = client.chat.completions.create(
            # Using a model that is good with JSON and following instructions
            model="gpt-3.5-turbo-1106", 
            messages=[
                {"role": "system", "content": "You are an expert quiz creator for educators."},
                {"role": "user", "content": prompt}
            ],
            # This ensures the model outputs a valid JSON object
            response_format={"type": "json_object"} 
        )
        response_content = completion.choices[0].message.content
        # The response from the model should already be a valid JSON string
        return json.loads(response_content)

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        raise ConnectionAbortedError(f"Failed to get a valid response from the AI model: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON from LLM response")
        raise ValueError("The AI model returned a response that was not valid JSON.")


@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_endpoint():
    """
    The main API endpoint to handle PDF upload and quiz generation.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type, please upload a PDF"}), 400

    try:
        # Read the file content into memory
        pdf_bytes = file.read()
        
        # Extract text from the PDF using PyMuPDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        
        if not text.strip():
            return jsonify({"error": "Could not extract any text from the PDF."}), 400

        # Generate the quiz using the extracted text
        quiz_data = generate_quiz_from_text(text)
        
        # Validate that the response has the expected keys
        if "multiple_choice" not in quiz_data or "true_false" not in quiz_data:
            raise ValueError("The generated quiz is missing required sections.")

        return jsonify(quiz_data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Return a generic server error for security
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Runs the Flask app on localhost, port 5000
    app.run(debug=True)