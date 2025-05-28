
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
import torch
from generate import generate_route, decode_holds, drawClimb
from setter import Setter
import datetime
import os
import logging
import sys
from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define required resources
required_resources = {
    "kilterboardImg.jpg": os.path.join(os.path.dirname(__file__), "kilterboardImg.jpg"),
    "model_file": os.path.join(os.path.dirname(__file__), "kilter_setter_epoch_9.pt")
}

# Ensure the climbs directory exists
climbs_dir = os.path.join(os.path.dirname(__file__), "climbs")
if not os.path.exists(climbs_dir):
    os.makedirs(climbs_dir)

# Validate required resources
missing_resources = []
for resource_name, resource_path in required_resources.items():
    if not os.path.exists(resource_path):
        missing_resources.append(resource_name)
        logging.error(f"Missing required resource: {resource_name} at {resource_path}")

# Check alternative locations for kilterboard image
if "kilterboardImg.jpg" in missing_resources:
    alt_paths = [
        os.path.join(os.path.dirname(__file__), "mysite", "kilterboardImg.jpg"),
        "kilterboardImg.jpg"
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            logging.info(f"Found kilterboardImg.jpg at alternative location: {path}")
            missing_resources.remove("kilterboardImg.jpg")
            break

# Set global flag for resource availability
resources_available = len(missing_resources) == 0

if not resources_available:
    logging.error("Missing required resources. Application may not function correctly.")
    logging.error(f"Missing: {', '.join(missing_resources)}")
    logging.error("Please ensure all required files are available before continuing.")

@app.route('/')
def hello_world():
    return 'Hello from Flask yippee!'

@app.route('/test')
def test():
    return 'hello from test yippee'

@app.route('/generate', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        # Check if all required resources are available
        if not resources_available:
            error_message = f"Server is missing required resources: {', '.join(missing_resources)}"
            logging.error(error_message)
            return jsonify({
                'error': error_message,
                'details': "Please ensure all required files are available.",
                'missing_files': missing_resources
            }), 503  # Service Unavailable

        data = request.get_json()
        grade = data.get('grade')
        angle = data.get('angle')

        if grade is None or angle is None:
            return jsonify({'error': 'Missing grade or angle parameter.'}), 400

        try:
            model = Setter(vocab_size=16000).to('cpu')
            # Fix model path to use the correct local path
            model_path = os.path.join(os.path.dirname(__file__), "kilter_setter_epoch_9.pt")
            model.load_state_dict(torch.load(model_path))

            tokens = generate_route(model, grade=grade, angle=angle)
            climb = decode_holds(tokens)

            img = drawClimb(climb)

            imgName = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_climb.png"
            
            # Use proper path joining
            img_path = os.path.join(climbs_dir, imgName)
            img.save(img_path)
            
            return send_file(img_path)
        except Exception as e:
            logging.error(f"Error generating climb: {str(e)}", exc_info=True)
            return jsonify({'error': f"Server error: {str(e)}"}), 500
    else:
        return 'generate this is'

@app.route('/status')
def status():
    """Endpoint to check server status and required resources"""
    if resources_available:
        return jsonify({
            'status': 'ok',
            'message': 'Server is running and all required resources are available.'
        })
    else:
        return jsonify({
            'status': 'warning',
            'message': 'Server is running but missing required resources.',
            'missing_resources': missing_resources
        }), 503  # Service Unavailable

if __name__ == '__main__':
    # Print status message
    if not resources_available:
        print("\n*** WARNING: MISSING REQUIRED RESOURCES ***")
        print(f"The following required files are missing: {', '.join(missing_resources)}")
        print("The application may not function correctly without these files.")
        print("Please ensure all required files are available in the correct locations.")
        print("You can check the '/status' endpoint for more information.\n")
        
    app.run(debug=True)
