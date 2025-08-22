from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import shutil
from pathlib import Path  # Add this import

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load YOLO model (ensure 'best.pt' is in the root folder)
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    results_img_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)

            # Run YOLO detection — always save to a temp folder
            results = model.predict(
                source=upload_path,
                save=True,
                project='static/results',
                name='predict',  # YOLO will save inside static/results/predict
                exist_ok=True
            )

            # Find the saved image path
            saved_dir = Path(results[0].save_dir)  # Convert to Path object
            detected_files = list(saved_dir.glob("*"))

            if detected_files:
                detected_file = detected_files[0]
                # Move file to static/results root
                final_path = os.path.join(app.config['RESULTS_FOLDER'], detected_file.name)
                shutil.move(str(detected_file), final_path)

                results_img_path = f"results/{detected_file.name}"

    return render_template('index.html', results_img_path=results_img_path)

# @app.route('/static/<path:filename>')
# def static_files(filename):
#     return send_from_directory('static', filename)
@app.route('/debug', methods=['GET'])
def debug():
    upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
    results_files = os.listdir(app.config['RESULTS_FOLDER'])
    return {
        'upload_files': upload_files,
        'results_files': results_files,
        'upload_path': app.config['UPLOAD_FOLDER'],
        'results_path': app.config['RESULTS_FOLDER']
    }

# Uncomment and modify this route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Add a new route to list static files
@app.route('/static-files')
def list_static_files():
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    results = os.listdir(app.config['RESULTS_FOLDER'])
    
    files = {
        'uploads': [f'/static/uploads/{f}' for f in uploads],
        'results': [f'/static/results/{f}' for f in results]
    }
    
    # Return a simple HTML page listing the files
    html = '<h2>Static Files:</h2>'
    html += '<h3>Uploads:</h3><ul>'
    for f in files['uploads']:
        html += f'<li><a href="{f}" target="_blank">{f}</a></li>'
    html += '</h3><h3>Results:</h3><ul>'
    for f in files['results']:
        html += f'<li><a href="{f}" target="_blank">{f}</a></li>'
    html += '</ul>'
    
    return html
if __name__ in "__main__":
    app.run(host="0.0.0.0", port=5000,debug=False)
