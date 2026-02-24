import os
import logging
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify

import match_engine

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key-change-me')
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

@app.context_processor
def inject_current_year():
    return dict(current_year=datetime.now().year)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_resumes():
    try:
        job_desc = request.form.get('description')
        if not job_desc or not job_desc.strip():
            raise ValueError("Job or project description is required.")

        resume_files = request.files.getlist('resumes')
        app.logger.info(f"Received {len(resume_files)} files from frontend")

        valid_files = []
        for file in resume_files:
            if file and file.filename and allowed_file(file.filename):
                valid_files.append(file)
            else:
                app.logger.warning(f"Skipped invalid file: filename={file.filename if file else 'None'}")

        if not valid_files:
            raise ValueError("At least one valid resume file (PDF/DOCX) is required.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            resume_paths = []
            for file in valid_files:
                filename = secure_filename(file.filename)
                path = os.path.join(tmpdirname, filename)
                file.save(path)
                resume_paths.append(path)
                app.logger.info(f"Saved: {filename} ({os.path.getsize(path)} bytes)")

            if not resume_paths:
                raise ValueError("No files could be saved.")

            app.logger.info(f"Processing {len(resume_paths)} valid resumes")

            results = match_engine.get_top_matches(job_desc, resume_paths)
            top_10 = results[:10]

            return render_template('results.html', results=top_10)

    except ValueError as ve:
        app.logger.error(f"Validation error: {ve}")
        flash(str(ve), 'danger')
        return redirect(url_for('index'))

    except Exception as e:
        app.logger.exception("Unexpected error in /match")
        flash("An unexpected error occurred. Please try again.", 'danger')
        return redirect(url_for('index'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    debug = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=5000)