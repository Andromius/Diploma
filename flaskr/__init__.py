import os
import cv2

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from logging import Logger
from pipeline.pipeline_builder import PipelineCreator
from base64 import b64encode
import numpy as np

UPLOAD_FOLDER = 'uploads'  # Directory where uploaded files will be stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Allowed image extensions

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    # app.config.from_mapping(
    #     SECRET_KEY='dev',
    #     DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    # )
    if test_config is None:
        settings = os.environ['APP_SETTINGS']
        # load the instance config, if it exists, when not testing
        app.config.from_object(f'flaskr.{settings}')
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    pipelineCreator = PipelineCreator(app.logger)
    pipeline = pipelineCreator.construct_graffiti("maskRCNN")
    
    @app.route('/upload', methods=['POST'])
    def upload_image():
        """Handles the image upload process."""
        app.logger.info('Upload image endpoint called')
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part in the request!', 'error')
                return redirect(request.url) # Redirect back to the upload form (which is the referrer, or index)

            file = request.files['file']

            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file!', 'warning')
                return redirect(url_for('index')) # Redirect to index page

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename) # Sanitize the filename
                
                # Create the uploads directory if it doesn't exist
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                    
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(file_path)
                    flash(f'Image "{filename}" uploaded successfully!', 'success')

                    image = cv2.imread(file_path)
                    if image is None:
                        flash(f"Error loading the image: {filename}", 'error')
                        return redirect(url_for('index'))

                    data = pipeline.execute(image)
                    result = [data["image"]]
                    result = result + data["final_images"]
                    flash(f"Number of images generated: {len(result)}", 'success')
                    app.logger.info(f"Number of images generated: {len(result)}")

                    base64_images = []
                    for i, img_binary in enumerate(result):
                        try:
                            # If img_binary is already a numpy array
                            if isinstance(img_binary, np.ndarray):
                                # Convert numpy array to binary jpg
                                _, buffer = cv2.imencode('.jpg', img_binary)
                                img_binary = buffer.tobytes()
                            
                            # Convert binary to base64
                            img_base64 = b64encode(img_binary).decode('utf-8')
                            
                            # Determine content type (assuming JPEG for simplicity)
                            content_type = 'image/jpeg'
                            
                            # Create the data URL
                            data_url = f'data:{content_type};base64,{img_base64}'
                            base64_images.append(data_url)
                        except Exception as e:
                            app.logger.error(f"Error converting image {i}: {str(e)}")
                            # Continue with other images even if one fails
                    
                    return render_template("results.html", user_images=base64_images)
                except Exception as e:
                    flash(f'An error occurred while saving the file: {e}', 'error')
                    return redirect(url_for('index'))
            else:
                flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.', 'error')
                return redirect(url_for('index'))

        # If GET request or other issues, redirect to index
        return redirect(url_for('index'))

    @app.route('/')
    def index():
        # Function to handle home page
        return render_template('index.html')

    return app