# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**setter** is a transformer-based neural network designed to generate rock climbing routes on the Kilterboard. The model takes a desired grade (V-scale) and angle as input and generates a route on the 12x12 Kilterboard.

The project consists of:
- A PyTorch transformer model (`setter.py`) trained on Kilterboard climbing data
- Training and inference scripts
- A Flask backend API (`flask_stuff/main.py`) hosted on PythonAnywhere
- A React frontend (`setter_app/`) deployed to GitHub Pages

## Development Commands

### Python/ML Backend

```bash
# Install Python dependencies (requires Python 3.13)
pip install -r requirements.txt

# Train the model (generates kilter_setter_epoch_*.pt files)
python training.py

# Generate a climbing route (standalone)
python generate.py <grade> <angle>
# Example: python generate.py 5 40

# Run Flask server locally (for development)
cd flask_stuff
python main.py

# Deploy Flask app to PythonAnywhere
# (Follow PythonAnywhere deployment process, using wsgi.py)
```

### React Frontend

```bash
cd setter_app

# Install dependencies
npm install

# Run development server (localhost:5173)
npm run dev

# Build for production
npm run build

# Lint code
npm run lint

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## Architecture

### Model Architecture

The core model is a **Transformer-based sequence generator** (`setter.py`):

- **Embedding Layer**: Embeds hold tokens (encoded as `hold_id * 10 + hold_type`)
- **Conditioning Embeddings**: Separate linear layers for grade and angle, added to sequence embeddings
- **Transformer**: 6-layer encoder-decoder with 8 attention heads and 256-dimensional embeddings
- **Output**: Predicts next hold token autoregressively

**Token Encoding Scheme**:
- Hold tokens: `hold_id * 10 + TYPE_TO_TOKEN[hold_type]`
- Special tokens: START (4), END (5), PAD (6)
- Hold types: Start (12→0), Handholds (13→1), Finish (14→2), Footholds (15→3)

**Training Process**:
- Uses teacher forcing with next-token prediction
- Dataset: `ilsenatorov/kilterboard` from HuggingFace
- Normalizes grade as `(grade - 10) / 21.0`, angle as `angle / 70.0`
- Trains for 10 epochs, saves checkpoint after each epoch

### Data Flow

1. **Preprocessing** (`preprocessing.py`):
   - Loads Kilterboard dataset from HuggingFace
   - Parses hold sequences from text format (e.g., "p1081r15")
   - Tokenizes holds using custom encoding scheme
   - Applies random truncation for next-token prediction training

2. **Generation** (`generate.py`):
   - Loads trained model checkpoint (default: `kilter_setter_epoch_9.pt`)
   - Generates route autoregressively with temperature sampling
   - Decodes tokens back to hold positions
   - Visualizes route on Kilterboard image using PIL

3. **Flask Backend** (`flask_stuff/main.py`):
   - `/generate` endpoint accepts POST with `{grade, angle}`
   - Loads model, generates route, renders image
   - Returns PNG image to client
   - Includes resource validation and error logging

4. **React Frontend** (`setter_app/src/App.jsx`):
   - Simple UI with grade/angle selectors
   - Sends POST request to PythonAnywhere backend
   - Displays generated climbing route image
   - Shows loading spinner during generation

### Frontend Structure

- **Vite + React 19** setup with HMR
- **Single-page app** with all logic in `App.jsx`
- **Deployment**: Uses `gh-pages` to deploy to GitHub Pages
- **Base path**: Configured as `/setter` in `vite.config.js`
- **API endpoint**: `https://BaritoDespacito.pythonanywhere.com/generate`

### Key Files

- `setter.py` - Transformer model definition
- `training.py` - Model training loop
- `generate.py` - Route generation and visualization
- `preprocessing.py` - Data loading and tokenization
- `flask_stuff/main.py` - Flask API server
- `flask_stuff/setter.py` - Model definition (duplicate for deployment)
- `flask_stuff/generate.py` - Generation utilities (duplicate for deployment)
- `setter_app/src/App.jsx` - React frontend
- `kilterboardImg.jpg` - Base image for route visualization

## Hold Coordinate Mapping

The `drawClimb()` function maps hold IDs to pixel coordinates:

- **Bolt-ons** (1090-1395): 17-column grid, 71px spacing
- **Kickboard bolt-ons** (1073-1089): Single row at bottom
- **Kickboard footholds** (1448-1465): 18-column row
- **Regular footholds** (1466-1600): 9-column grid with offset alternation

Hold types are rendered with specific colors (lime=start, cyan=handholds, fuchsia=finish, orange=footholds).

## Important Notes

- The model checkpoints (`kilter_setter_epoch_*.pt`) are 77MB each and committed to the repo
- Flask deployment requires `kilterboardImg.jpg` and model checkpoint in `flask_stuff/`
- Frontend expects backend to return a PNG blob, not JSON
- The vocab size is set to 16000 (max hold_id ~1600, with safety margin)
