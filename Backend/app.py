from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Dynamically import PDFStorageManager
data_extraction_path = os.path.join(project_root, 'Data Extraction')
sys.path.insert(0, data_extraction_path)

from Backend.pdf_storage_manager import PDFStorageManager

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'Frontend', 'templates'))
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize PDF Storage Manager
pdf_storage = PDFStorageManager()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Additional metadata from form
        metadata = {
            'original_filename': filename,
            'category': request.form.get('category', 'Uncategorized'),
            'description': request.form.get('description', '')
        }
        
        try:
            # Store PDF in MongoDB
            file_id = pdf_storage.store_pdf(filepath, metadata)
            
            # Optional: Remove temporary file after storage
            os.remove(filepath)
            
            return jsonify({
                'message': 'PDF uploaded successfully', 
                'file_id': str(file_id)
            }), 200
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/search', methods=['GET'])
def search_pdfs():
    # Example search endpoint
    query = {}
    category = request.args.get('category')
    if category:
        query['category'] = category
    
    results = pdf_storage.search_pdfs(query)
    return jsonify(results), 200

@app.route('/download/<file_id>', methods=['GET'])
def download_pdf(file_id):
    try:
        pdf_data = pdf_storage.retrieve_pdf(file_id)
        if pdf_data:
            # Implement download logic
            return send_file(
                pdf_data['file'], 
                mimetype='application/pdf', 
                as_attachment=True, 
                download_name=pdf_data['metadata']['filename']
            )
        else:
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)