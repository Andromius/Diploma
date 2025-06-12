import os
import cv2

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from logging import Logger
from pipeline.pipeline_builder import PipelineCreator
from base64 import b64encode
from . import db
import numpy as np
import hashlib

UPLOAD_FOLDER = 'uploads'  # Directory where uploaded files will be stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Allowed image extensions

def hash_file_md5(file):
    hasher = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b''):
        hasher.update(chunk)
    file.seek(0)
    return hasher.hexdigest()

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

    settings = os.environ['APP_SETTINGS']
    # load the instance config, if it exists, when not testing
    app.config.from_object(f'flaskr.{settings}')

    db.initialize_db(app.config['DATABASE'])

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
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
                filename = hash_file_md5(file)
                if db.exists_image_name(db.get_db_connection(app.config['DATABASE']), filename, app.logger):
                    flash('This image has already been uploaded!', 'warning')
                    return redirect(url_for('index'))

                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                    flash('This image has already been uploaded!', 'warning')
                    return redirect(url_for('index'))
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(file_path)
                    flash(f'Image "{secure_filename(file.filename)}" uploaded successfully!', 'success')
                    image = cv2.imread(file_path)
                    if image is None:
                        flash(f"Error loading the image: {filename}", 'error')
                        return redirect(url_for('index'))
                    
                    connection = db.get_db_connection(app.config['DATABASE'])
                    pipelineCreator = PipelineCreator(app.logger, app.config['RESOURCES_PATH'], connection)
                    pipeline = pipelineCreator.construct_graffiti("maskRCNN")

                    app.logger.info(f"Executing pipeline for image: {filename}")
                    data = pipeline.execute(image, filename)
                    result = [data["image"]]
                    result = result + data["final_images"]
                    flash(f"Number of images generated: {len(result)}", 'success')
                    app.logger.info(f"Number of images generated: {len(result)}")
                    
                    db.close_db_connection(connection)
                    base64_images = []
                    for i, img_binary in enumerate(result):
                        try:
                            if isinstance(img_binary, np.ndarray):
                                # Convert numpy array to binary jpg
                                _, buffer = cv2.imencode('.jpg', img_binary)
                                img_binary = buffer.tobytes()
                            
                            img_base64 = b64encode(img_binary).decode('utf-8')
                            content_type = 'image/jpeg'
                            
                            data_url = f'data:{content_type};base64,{img_base64}'
                            base64_images.append(data_url)
                        except Exception as e:
                            app.logger.error(f"Error converting image {i}: {str(e)}")
                    
                    base64_similar_images = []
                    for i, sim_images in enumerate(data['similar_images']):
                        sim_base64_images = []
                        for j, sim_image in enumerate(sim_images):
                            try:
                                if isinstance(sim_image, np.ndarray):
                                    _, buffer = cv2.imencode('.jpg', sim_image)
                                    sim_image = buffer.tobytes()
                                sim_img_base64 = b64encode(sim_image).decode('utf-8')
                                data_url = f'data:image/jpeg;base64,{sim_img_base64}'
                                sim_base64_images.append(data_url)
                            except Exception as e:
                                app.logger.error(f"Error converting similar image {i}-{j}: {str(e)}")
                        base64_similar_images.append(sim_base64_images)
                    return render_template("results.html",
                                            user_images=base64_images, colorfulnesses=data['colorfulness_data'],
                                            similar_images=base64_similar_images)
                except Exception as e:
                    db.close_db_connection(connection)
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