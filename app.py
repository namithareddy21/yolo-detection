# app.py

from flask import Flask, render_template
import os

# Render sets the PORT environment variable. We use it if available, 
# otherwise default to 5000 for local development.
PORT = os.environ.get('PORT', 5000)

# Initialize the Flask application
# It's configured to look for:
# - Templates in the 'templates/' folder
# - Static files (like yolov8n.onnx) in the 'static/' folder
app = Flask(__name__)

@app.route('/')
def index():
    """
    Renders the main HTML page.
    """
    return render_template('index.html')

if __name__ == '__main__':
    # Running locally uses Flask's built-in server.
    # In a production environment like Render, Gunicorn (defined in Procfile) is used.
    print(f"Starting Flask server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
