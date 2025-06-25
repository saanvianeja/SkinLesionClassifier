from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from enhanced_skin_analysis import predict_lesion, enhanced_analyzer
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-for-dev")

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Fitzpatrick skin types
FITZPATRICK_TYPES = {
    'I': 'Type I - Very Fair (Always burns, never tans)',
    'II': 'Type II - Fair (Usually burns, tans minimally)', 
    'III': 'Type III - Medium (Sometimes burns, tans uniformly)',
    'IV': 'Type IV - Olive (Rarely burns, tans easily)',
    'V': 'Type V - Dark (Very rarely burns, tans very easily)',
    'VI': 'Type VI - Very Dark (Never burns, tans very easily)'
}

def cleanup_old_uploads():
    """Clean up uploaded files older than 1 hour"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            current_time = time.time()
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
    except Exception as e:
        app.logger.warning(f"Cleanup failed: {e}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    return {'status': 'healthy', 'message': 'Skin Lesion Classifier is running'}, 200

@app.route('/', methods=['GET', 'POST'])
def home():
    """Main route for file upload and prediction"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'image' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['image']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check file extension
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload a valid image file.', 'error')
                return redirect(request.url)
            
            # Get form data
            skin_type = request.form.get('skin_type', 'III')
            body_part = request.form.get('body_part', 'other')
            if not body_part or body_part == '' or body_part == 'None':
                body_part = 'other'
            has_evolved = 'has_evolved' in request.form
            
            # Handle evolution_weeks with proper validation
            evolution_weeks_raw = request.form.get('evolution_weeks', '')
            if has_evolved and evolution_weeks_raw and evolution_weeks_raw.strip():
                try:
                    evolution_weeks = int(evolution_weeks_raw)
                except ValueError:
                    evolution_weeks = 0
            else:
                evolution_weeks = 0
            
            # Get additional metadata
            try:
                age = int(request.form.get('age', 50))
            except Exception:
                age = 50
            try:
                uv_exposure = int(request.form.get('uv_exposure', 5))
            except Exception:
                uv_exposure = 5
            family_history = 'family_history' in request.form
            manual_length = request.form.get('manual_length')
            manual_width = request.form.get('manual_width')
            
            # Convert manual measurements to float if provided
            if manual_length:
                try:
                    manual_length = float(manual_length)
                except ValueError:
                    manual_length = None
            if manual_width:
                try:
                    manual_width = float(manual_width)
                except ValueError:
                    manual_width = None
            
            # Save uploaded file
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = str(int(time.time()))
                unique_filename = f"{timestamp}_{filename}"
                
                # Create upload directory if it doesn't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                app.logger.info(f"File saved to: {filepath}")
                
                # Detect skin tone from image
                detected_skin_type = enhanced_analyzer.detect_skin_type(filepath)
                skin_type = detected_skin_type  # Override form value with detected
                
                # Make enhanced prediction
                try:
                    prediction, confidence, analysis_data = predict_lesion(
                        filepath, skin_type, body_part, has_evolved, evolution_weeks,
                        manual_length, manual_width, age, uv_exposure, family_history
                    )
                    
                    app.logger.info(f"Enhanced Prediction: {prediction}, Confidence: {confidence}%")
                    
                    # Generate comprehensive analysis summary
                    analysis_summary = None
                    if analysis_data:
                        # Defensive: fill in missing keys with defaults
                        analysis_summary = {
                            'ABCDE_feature_analysis': analysis_data.get('ABCDE_feature_analysis', {}),
                            'cnn_analysis': analysis_data.get('cnn_analysis', {}),
                            'metadata_risk_analysis': analysis_data.get('metadata_risk_analysis', {}),
                            'combined_score': analysis_data.get('combined_score', 0.0),
                            'combined_score_explanation': analysis_data.get('combined_score_explanation', ''),
                            'detected_skin_tone': analysis_data.get('detected_skin_tone', f'Type {skin_type}'),
                            'analysis_type': analysis_data.get('analysis_type', 'enhanced'),
                            'skin_type_adjustments': analysis_data.get('skin_type_adjustments', {}),
                            'manual_measurements': analysis_data.get('manual_measurements', {})
                        }
                    
                    # Clean up old files
                    cleanup_old_uploads()
                    
                    return render_template('index.html', 
                                         result=prediction, 
                                         confidence=confidence, 
                                         image_path=filepath,
                                         filename=unique_filename,
                                         skin_type=skin_type,
                                         skin_type_description=FITZPATRICK_TYPES[skin_type],
                                         analysis_summary=analysis_summary,
                                         detected_skin_tone=f'Type {skin_type}',
                                         fitzpatrick_types=FITZPATRICK_TYPES,
                                         body_part_options=['face', 'scalp', 'neck', 'chest', 'back', 'abdomen', 'arms', 'hands', 'legs', 'feet', 'other'],
                                         age=age,
                                         uv_exposure=uv_exposure,
                                         family_history=family_history,
                                         manual_length=manual_length,
                                         manual_width=manual_width)
                    
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    flash('Error processing image. Please try again with a different image.', 'error')
                    return redirect(request.url)
                    
        except Exception as e:
            app.logger.error(f"Upload error: {str(e)}")
            flash('An error occurred while processing your upload. Please try again.', 'error')
            return redirect(request.url)
    
    # GET request - show the form
    body_part_options = ['face', 'scalp', 'neck', 'chest', 'back', 'abdomen', 'arms', 'hands', 'legs', 'feet', 'other']
    return render_template('index.html', fitzpatrick_types=FITZPATRICK_TYPES, body_part_options=body_part_options)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    app.logger.error(f"Internal error: {str(e)}")
    return render_template('index.html', error="Internal server error"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exceptions"""
    app.logger.error(f"Unhandled exception: {str(e)}")
    return render_template('index.html', error="An unexpected error occurred"), 500

if __name__ == '__main__':
    # Log startup information
    app.logger.info(f"Starting Flask app on port 5000")
    app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Template folder: {app.template_folder}")
    app.logger.info(f"Static folder: {app.static_folder}")
    
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)