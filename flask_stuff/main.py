from flask import Flask, request, jsonify
import torch
from generate import generate_route, decode_holds
from setter import Setter

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':

        data = request.get_json()
        grade = data.get('grade')
        angle = data.get('angle')

        if grade is None or angle is None:
            return jsonify({'error': 'Missing grade or angle parameter.'}), 400

        try:
            model = Setter(vocab_size=16000).to('cpu')
            model.load_state_dict(torch.load("../kilter_setter_epoch_9.pt"))

            tokens = generate_route(model, grade=grade, angle=angle)
            climb = decode_holds(tokens)
            return jsonify({
                'climb': climb,
                'tokens': tokens
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500