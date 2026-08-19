FROM python:3.13-slim

WORKDIR /app

# Torch's default PyPI wheel bundles CUDA support (multi-GB); this container only ever
# does CPU inference (the model is small - a few seconds of generation time on a
# laptop CPU), so install the much smaller CPU-only build explicitly first. The
# requirements.txt install below then just confirms the torch>=2.0.0 constraint is
# already satisfied rather than pulling a second, different build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (default 8080) and requires the container to listen on it.
ENV PORT=8080
EXPOSE 8080

CMD waitress-serve --threads=4 --host=0.0.0.0 --port=${PORT} main:app
