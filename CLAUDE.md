# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**setter** is a transformer-based neural network designed to generate rock climbing routes on the Kilterboard. The model takes a desired grade (V-scale) and angle as input and generates a route on the 12x12 Kilterboard.

The project consists of:
- A PyTorch transformer model (`setter.py`) trained on Kilterboard climbing data, plus a `RouteCritic` (`critic.py`) that judges/reranks generated routes by predicted difficulty
- Training and inference scripts, and `evaluate.py` for automated quality tracking (`eval_history.jsonl`)
- A Flask backend API (`flask_stuff/main.py`) deployed to Google Cloud Run
- An Expo (React Native + web) frontend (`app/`) deployed to Vercel, with Supabase for accounts/saved routes/ratings

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

# Deploy Flask app to Cloud Run
cd flask_stuff
gcloud run deploy setter-api --source . --region us-central1 --allow-unauthenticated

# Track model quality over time (also runs automatically at the end of training.py)
python evaluate.py [checkpoint] --label "..." [--no-report] [--no-critic] [--history]
```

### Frontend (Expo)

```bash
cd app

# Install dependencies
npm install

# Run development server (web)
npm run web

# Build static web export
npm run build:web

# Deploy to Vercel (production)
npx vercel --prod
```

Env vars (see `app/.env.example`): `EXPO_PUBLIC_API_URL` (Cloud Run backend),
`EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY`. Set these in the Vercel
project's environment variables for production, or in `app/.env` for local dev.
The Supabase schema (`supabase/schema.sql`) must be run once in the Supabase project's
SQL editor before accounts/saved routes/ratings will work.

## Architecture

### Model Architecture

The core model is a **Transformer-based sequence generator** (`setter.py`):

- **Embedding Layer**: Embeds hold tokens (encoded as `hold_id * 10 + hold_type`), plus real (x, y) board-coordinate embeddings added to each token
- **Conditioning**: Grade + angle → MLP → single vector, used both as cross-attention memory and broadcast additively into every position
- **Transformer**: decoder-only, 8-layer, 8 attention heads, 384-dimensional embeddings (~31M params)
- **Auxiliary heads**: `predict_length`, `predict_foot_fraction`, `predict_spacing`, trained off the conditioning vector via MSE loss
- **Decoding**: nucleus (top-p) sampling with graph-adjacency-constrained logits (holds must be reachable from placed structure), hard duplicate-hold masking, best-of-N candidates reranked by `RouteCritic`
- **Output**: Predicts next hold token autoregressively

**Token Encoding Scheme**:
- Hold tokens: `hold_id * 10 + TYPE_TO_TOKEN[hold_type]`
- Special tokens: START (4), END (5), PAD (6)
- Hold types: Start (12→0), Handholds (13→1), Finish (14→2), Footholds (15→3)

**Training Process**:
- Uses teacher forcing with next-token prediction, holds sequenced bottom-to-top by real height
- Dataset: merged from `ilsenatorov/kilterboard`, `gabriead/Kilterboard`, and `Vilin97/KilterBoard` (~256k+ unique routes)
- Public grade (V-scale) is converted to raw dataset difficulty via `V_GRADE_TO_DIFFICULTY` before normalizing; angle normalized as `angle / 70.0`
- Trains up to 40 epochs (AdamW, label smoothing, warm-start from previous best checkpoint), saves best/latest checkpoints and runs `evaluate.py` automatically at the end

**RouteCritic** (`critic.py`): a separate bidirectional transformer encoder that judges a *finished* route and predicts its difficulty, used to rerank generation candidates and shown in evaluation reports.

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

3. **Flask Backend** (`flask_stuff/main.py`), deployed to Cloud Run:
   - `/generate` endpoint accepts POST with `{grade, angle}`, loads model + critic once at startup, generates and reranks a route, renders image, returns PNG
   - `/changelog` endpoint serves `eval_history.jsonl` as JSON for the frontend's changelog page
   - `/status` endpoint reports resource-loading health
   - Includes resource validation and error logging

4. **Expo Frontend** (`app/`):
   - Grade/angle picker → POST to Cloud Run `/generate` → displays the returned route image
   - Changelog tab fetches `/changelog` and renders model quality trends over time
   - Saved/Profile tabs use Supabase for auth, saved routes, and ratings (no-op until Supabase env vars are set)

### Frontend Structure

- **Expo (React Native + react-native-web)** with `expo-router` file-based routing, targeting web now (mobile later)
- Routes: `app/(tabs)/index.tsx` (Generate), `saved.tsx`, `changelog.tsx`, `profile.tsx`, plus `app/login.tsx`
- Shared logic in `src/lib/`: `api.ts` (Cloud Run client), `supabase.ts` / `auth.tsx` (Supabase client + auth context), `routes.ts` (saved routes/ratings), `config.ts`, `theme.ts`
- **Deployment**: static web export (`expo export --platform web`) deployed to Vercel via `vercel.json` (`buildCommand`/`outputDirectory`)
- **Backend URL**: `https://setter-api-490491172314.us-central1.run.app` (Cloud Run)

### Key Files

- `setter.py` - Transformer model definition
- `critic.py` - RouteCritic model definition
- `training.py` / `train_critic.py` - Model/critic training loops
- `generate.py` - Route generation and visualization
- `preprocessing.py` - Data loading and tokenization
- `evaluate.py` - Automated evaluation, HTML reports, `eval_history.jsonl` tracking
- `flask_stuff/main.py` - Flask API server (Cloud Run)
- `flask_stuff/setter.py`, `flask_stuff/generate.py`, `flask_stuff/critic.py` - duplicates for deployment
- `flask_stuff/eval_history.jsonl` - duplicate of root `eval_history.jsonl`, kept in sync manually for the `/changelog` endpoint
- `app/` - Expo frontend
- `supabase/schema.sql` - Supabase schema (profiles, routes, ratings) - run once in the Supabase SQL editor
- `kilterboardImg.jpg` - Base image for route visualization

## Hold Coordinate Mapping

The `drawClimb()` function maps hold IDs to pixel coordinates:

- **Bolt-ons** (1090-1395): 17-column grid, 71px spacing
- **Kickboard bolt-ons** (1073-1089): Single row at bottom
- **Kickboard footholds** (1448-1465): 18-column row
- **Regular footholds** (1466-1600): 9-column grid with offset alternation

Hold types are rendered with specific colors (lime=start, cyan=handholds, fuchsia=finish, orange=footholds).

## Important Notes

- Model checkpoints (`kilter_setter_best.pt`, `critic_best.pt`) are tracked with Git LFS (see `.gitattributes`) since the scaled-up model exceeds GitHub's 100MB limit
- Flask deployment requires `kilterboardImg.jpg` and both checkpoints in `flask_stuff/`
- Frontend expects `/generate` to return a PNG blob, not JSON
- `VOCAB_SIZE = 16020` (max hold_id ~1600, with safety margin)
