from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from generate import generate_route, decode_holds, drawClimb, MIN_GRADE, MAX_GRADE, MIN_ANGLE, MAX_ANGLE
from setter import Setter, VOCAB_SIZE, load_checkpoint_state_dict
from critic import load_critic
import datetime
import json
import os
import logging

CHANGELOG_PATH = os.path.join(os.path.dirname(__file__), "eval_history.jsonl")

app = Flask(__name__)
CORS(app)

# Sized to the --cpu=2 Cloud Run allocation (see CLAUDE.md deploy command). Intra-op
# parallelism should roughly match vCPU count; interop stays at 1 since this app never
# runs concurrent independent torch graphs per process - waitress handles request
# concurrency, not torch.
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define required resources
required_resources = {
    "kilterboardImg.jpg": os.path.join(os.path.dirname(__file__), "kilterboardImg.jpg"),
    "model_file": os.path.join(os.path.dirname(__file__), "kilter_setter_best.pt")
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

# Load the model once at startup rather than on every request - reloading a ~60MB
# checkpoint per request is what was making this endpoint trivial to overwhelm.
model = None
critic = None
if resources_available:
    device = "cpu"
    model = Setter(vocab_size=VOCAB_SIZE).to(device)
    model_path = required_resources["model_file"]
    model.load_state_dict(load_checkpoint_state_dict(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded and ready.")

    critic_path = os.path.join(os.path.dirname(__file__), "critic_best.pt")
    critic = load_critic(critic_path, device=device)
    logging.info("Critic loaded, using it to rerank candidates." if critic else
                 "No critic checkpoint found, falling back to hold-count proxy.")


@app.route('/')
def hello_world():
    return 'Hello from Flask yippee!'

@app.route('/test')
def test():
    return 'hello from test yippee'

@app.route('/generate', methods=['POST'])
def generate():
    if not resources_available or model is None:
        error_message = f"Server is missing required resources: {', '.join(missing_resources)}"
        logging.error(error_message)
        return jsonify({
            'error': error_message,
            'details': "Please ensure all required files are available.",
            'missing_files': missing_resources
        }), 503  # Service Unavailable

    data = request.get_json(silent=True) or {}
    grade = data.get('grade')
    angle = data.get('angle')

    if grade is None or angle is None:
        return jsonify({'error': 'Missing grade or angle parameter.'}), 400

    # Reject anything that isn't a plain integer (bool is an int subclass, exclude it too)
    # before it reaches the model, and enforce the trained grade/angle domain.
    if isinstance(grade, bool) or isinstance(angle, bool) or not isinstance(grade, int) or not isinstance(angle, int):
        return jsonify({'error': 'grade and angle must be integers.'}), 400

    if not (MIN_GRADE <= grade <= MAX_GRADE):
        return jsonify({'error': f'grade must be between {MIN_GRADE} and {MAX_GRADE}.'}), 400

    if not (MIN_ANGLE <= angle <= MAX_ANGLE):
        return jsonify({'error': f'angle must be between {MIN_ANGLE} and {MAX_ANGLE}.'}), 400

    try:
        tokens = generate_route(model, grade=grade, angle=angle, critic=critic)
        climb = decode_holds(tokens)

        img = drawClimb(climb)

        imgName = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_climb.png"
        img_path = os.path.join(climbs_dir, imgName)
        img.save(img_path)

        return send_file(img_path)
    except Exception as e:
        # Log the real error server-side, but don't leak internals to the client.
        logging.error(f"Error generating climb: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate climb.'}), 500

@app.route('/changelog')
def changelog():
    """Serves eval_history.jsonl (model quality over time) as JSON for the frontend."""
    if not os.path.exists(CHANGELOG_PATH):
        return jsonify({'entries': []})

    entries = []
    with open(CHANGELOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed line in eval_history.jsonl")

    entries.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return jsonify({'entries': entries})

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

    app.run(debug=False)
