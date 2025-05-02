FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir rasterio requests beautifulsoup4

WORKDIR /app

COPY pull_tiff.py ./scanner.py

ENTRYPOINT ["python", "scanner.py"]