# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

setter is a transformer-based neural network that generates rock climbing routes on the Kilterboard (12x12 layout). The model takes a desired grade (V-scale: V1-V17) and angle (0-70 degrees) as input and outputs a sequence of holds representing a complete climbing route.

## Architecture

### Core Model (`setter.py`)
- **Setter class**: Custom PyTorch Transformer (nn.Module) with encoder-decoder architecture
- **Input embeddings**: Combines hold token embeddings with grade and angle embeddings
- **Special tokens**: START (ID: 4), END (ID: 5), PAD (ID: 6)
- **Model parameters**: vocab_size=16000, d_model=256, nhead=8, num_layers=6
- **Hold encoding**: Each hold is encoded as `hold_id * 10 + TYPE_TO_TOKEN[hold_type]`
- **Hold types**: Start (12/green), Handholds (13/cyan), Finish (14/purple), Footholds (15/orange)

### Data Pipeline (`preprocessing.py`)
- **Dataset**: Uses HuggingFace `ilsenatorov/kilterboard` dataset
- **Tokenization**: Custom hold-based tokenization (not text-based)
- **Normalization**: Grade normalized as `(grade - 10) / 21.0`, angle as `angle / 70.0`
- **Training approach**: Teacher forcing with sequence truncation for next-hold prediction

### Generation (`generate.py`)
- **Inference**: Autoregressive generation with temperature-based sampling (default: 0.8)
- **Visualization**: Renders generated routes on kilterboardImg.jpg with colored circles
- **Hold coordinate mapping**: Complex pixel coordinate calculations for bolt-ons, footholds, and kickboard positions

### Flask Backend (`flask_stuff/`)
- **main.py**: Flask API with `/generate` endpoint (POST: generates route, returns PNG)
- **Resource validation**: Checks for kilterboardImg.jpg and model checkpoint before serving
- **Deployment**: Configured for PythonAnywhere (see Procfile, wsgi.py)

### React Frontend (`setter_app/`)
- **Framework**: React + Vite
- **UI**: Grade/angle selectors that POST to Flask API
- **Deployment**: GitHub Pages via `npm run deploy`
- **API endpoint**: https://BaritoDespacito.pythonanywhere.com/generate

## Commands

### Python/PyTorch (Model Development)

**Train the model:**
```bash
python training.py
```
- Trains for 10 epochs with batch size 32
- Saves checkpoints as `kilter_setter_epoch_{0-9}.pt`
- Requires GPU for reasonable training time

**Generate a route locally:**
```bash
python generate.py <grade> <angle>
# Example: python generate.py 5 40
```
- Generates and displays a climbing route
- Saves output as timestamped PNG in current directory
- Uses `kilter_setter_epoch_9.pt` by default

**Convert model to ONNX:**
```bash
python convert_to_onnx.py
```
- Exports to `setter_app/public/models/model.onnx`
- For potential browser-based inference

### Flask Backend

**Run Flask development server:**
```bash
cd flask_stuff
python main.py
```
- Runs on http://localhost:5000 with debug mode

**Run with Waitress (production):**
```bash
waitress-serve --host=0.0.0.0 --port=8080 wsgi:app
```

**Check server status:**
```bash
curl http://localhost:5000/status
```

### React Frontend

**Install dependencies:**
```bash
cd setter_app
npm install
```

**Run development server:**
```bash
cd setter_app
npm run dev
```

**Build for production:**
```bash
cd setter_app
npm run build
```

**Deploy to GitHub Pages:**
```bash
cd setter_app
npm run deploy
```

**Lint code:**
```bash
cd setter_app
npm run lint
```

## Key Implementation Details

### Model Training vs Inference Modes
The Setter model behaves differently during training and inference:
- **Training**: Uses teacher forcing with decoder input and causal masks
- **Inference**: Uses encoder output as both src and tgt (decoder-free generation)

### Hold Coordinate System
Kilterboard holds are mapped to pixel coordinates with distinct logic for:
- Bolt-ons (1090-1395): 17-column grid, 71px spacing
- Kickboard footholds (1448-1465): 18-column grid at bottom
- Regular footholds (1466-1600): 9-column offset grid
- Kickboard bolt-ons (1073-1089): Reverse x-coordinate mapping

### Data Flow
1. Training: HuggingFace dataset → parseRow() → custom tokenization → Setter model
2. Generation: Grade/angle inputs → model inference → token IDs → decode_holds() → drawClimb() → PNG image
3. Web app: React UI → Flask API → model generation → image response

## File Structure Notes

- Root-level Python files (`setter.py`, `generate.py`, etc.) are for standalone model training/generation
- `flask_stuff/` duplicates some Python files for Flask deployment context
- Multiple `kilter_setter_epoch_*.pt` checkpoints exist (epoch 9 is typically used for inference)
- `kilterboardImg.jpg` is the base Kilterboard template image (required for visualization)
