# Use a GDAL base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Set the working directory in the container
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install pip packages within the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files and directories
COPY src/ ./src/
COPY defs/ ./defs/
COPY antioquia_partitions_aschips_13d5705258769.geojson .
COPY app/ ./app/
COPY generate_vegetation_map.py . 

# Volume for data directory
VOLUME ["/app/data"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
